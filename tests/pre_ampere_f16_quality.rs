//! Quality gate for the pre-Ampere load-time bf16 -> f16 conversion (#1542).
//!
//! Volta (sm_70) and Turing (sm_75) have no bf16 ALU, no bf16 tensor-core MMA
//! atom, and no cuBLAS bf16 GEMM, so `bf16_to_f16_at_load` converts bf16
//! weights to f16 as the checkpoint is loaded. f16 carries a 5-bit exponent
//! against bf16's 8-bit one, so the conversion trades dynamic range for an
//! order of magnitude of throughput. That trade is only defensible if the
//! range actually lost is range the model never used, and this file is what
//! establishes that rather than assuming it.
//!
//! The comparison is run as two processes rather than two arms inside one, because
//! the policy is applied once at load and cannot be toggled on a loaded model:
//!
//! ```text
//! cargo test --test pre_ampere_f16_quality --release --features cuda -- \
//!     --ignored test_pre_ampere_f16_ppl --nocapture
//! MLXCEL_CUDA_F16_NORMALIZE=0 cargo test --test pre_ampere_f16_quality --release \
//!     --features cuda -- --ignored test_pre_ampere_f16_ppl --nocapture
//! ```
//!
//! Compare the two PPL tables. The gate is stated in `PPL_GATE_REL` below and
//! has to be applied by the reader across the two runs; a single run cannot
//! fail it, and prints its numbers for that comparison.
//!
//! Non-finite values are checked inside a single run and DO fail it. An f16
//! overflow in the residual stream or in pre-softmax logits shows up as an inf
//! or a NaN in the returned log-likelihoods, and that is a hard failure
//! regardless of what the other arm produced.

mod common;
use common::repo_model_dir;

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use mlxcel::{CxxGenerator, LanguageModel, load_model};
use mlxcel_core::cache::KVCacheMode;

/// Window sizes required by #1542. Short windows stress the per-token
/// distribution, long ones stress accumulation depth in the residual stream,
/// which is where an f16 exponent runs out first.
const PPL_WINDOWS: [usize; 3] = [128, 512, 2048];

/// Chunks per window size for a dense bf16 checkpoint.
const PPL_CHUNKS: usize = 16;

/// Chunks for a large quantized checkpoint. Fewer, because the bf16 control arm
/// on a 27B runs at roughly 122 ms per prefill token on this class of hardware,
/// which puts sixteen 2048-token windows over an hour for that arm alone.
const PPL_CHUNKS_LARGE: usize = 4;

/// Windows for a large quantized checkpoint on a 32 GB card.
///
/// The 2048 window is absent, and not because it was inconvenient: a
/// 2048-token forward pass on a 27B checkpoint exhausts the card during
/// `evaluate_loglikelihoods` and aborts with `cudaMallocAsync ... out of
/// memory`. It does so identically in both arms, which is what establishes the
/// cause as capacity rather than dtype, and the f16 arm holds strictly less
/// weight memory than the bf16 arm it fails alongside. Deep-context behavior at
/// this model size is therefore untested here and would need a larger card.
const PPL_WINDOWS_LARGE: [usize; 2] = [128, 512];

/// Relative PPL increase the f16 arm may show against the bf16 arm before the
/// conversion is not worth its throughput. Applied by the reader across two
/// runs; see the module comment.
#[allow(dead_code)]
const PPL_GATE_REL: f64 = 0.01;

/// Compute perplexity over non-overlapping windows of the wikitext-2 excerpt.
///
/// NLL is accumulated across windows and exponentiated once at the end.
/// Averaging per-window perplexities would weight short and long windows
/// equally, which is wrong; `turbo_kv_e2e.rs` documents the same reasoning.
///
/// Returns `(ppl, target_tokens, non_finite_count)`.
fn compute_ppl_at(
    model: &impl LanguageModel,
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    window_len: usize,
    chunks_wanted: usize,
) -> (f64, usize, usize) {
    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("wikitext2_excerpt.txt");
    let corpus = fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("wikitext2 corpus missing at {corpus_path:?}: {e}"));

    // No special tokens: a BOS repeated once per window would contribute a
    // near-zero-information position to every window and bias short windows most.
    let all_ids = tokenizer.encode(&corpus, false).expect("tokenize corpus");
    let ids: Vec<i32> = all_ids.iter().map(|&id| id as i32).collect();

    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), KVCacheMode::Fp16);
    let mut total_nll = 0.0_f64;
    let mut total_targets = 0_usize;
    let mut non_finite = 0_usize;

    let chunks = chunks_wanted.min(ids.len() / window_len);
    for chunk in 0..chunks {
        let window = &ids[chunk * window_len..(chunk + 1) * window_len];
        let logprobs = generator.evaluate_loglikelihoods(model, window);

        for (pos, &lp) in logprobs.iter().enumerate() {
            if !lp.is_finite() {
                // Report the first one with enough context to locate it, then
                // keep counting so the failure message states the extent.
                if non_finite == 0 {
                    eprintln!(
                        "  NON-FINITE log-likelihood: window {chunk}, position {pos}, value {lp}"
                    );
                }
                non_finite += 1;
                continue;
            }
            total_nll += -(lp as f64);
        }
        total_targets += logprobs.len();
    }

    assert!(
        total_targets > 0,
        "no target tokens at window {window_len}: corpus too short"
    );
    (
        (total_nll / total_targets as f64).exp(),
        total_targets,
        non_finite,
    )
}

fn run_ppl_sweep(model_dir_name: &str, chunks: usize, windows: &[usize]) {
    let model_dir = repo_model_dir(model_dir_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {model_dir_name}: not found at {}.\n\
             Fetch with: ./target/release/mlxcel download {model_dir_name}",
            model_dir.display()
        );
        return;
    }

    let arm = match std::env::var("MLXCEL_CUDA_F16_NORMALIZE").as_deref() {
        Ok("0") | Ok("false") | Ok("off") => "bf16 (conversion opted out)",
        _ => "f16 (default on pre-Ampere)",
    };
    eprintln!("\n=== {model_dir_name} :: {arm} ===");

    let (model, tokenizer) = load_model(&model_dir).expect("load model");

    let mut total_non_finite = 0_usize;
    for &window in windows {
        let t0 = Instant::now();
        let (ppl, targets, non_finite) = compute_ppl_at(&model, &tokenizer, window, chunks);
        total_non_finite += non_finite;
        eprintln!(
            "  window {window:>5}: PPL={ppl:>10.4}  targets={targets:>6}  \
             non_finite={non_finite}  ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
    }

    assert_eq!(
        total_non_finite, 0,
        "{model_dir_name} produced {total_non_finite} non-finite log-likelihoods under {arm}; \
         an f16 exponent overflow is a hard failure, not a quality regression"
    );
}

#[test]
#[ignore = "requires Meta-Llama-3.1-8B-Instruct-bf16 weights and a CUDA device — \
            run with --release --features cuda -- --ignored test_pre_ampere_f16_ppl --nocapture"]
fn test_pre_ampere_f16_ppl() {
    run_ppl_sweep(
        "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        PPL_CHUNKS,
        &PPL_WINDOWS,
    );
}

/// Second family: a large quantized checkpoint.
///
/// #1542 asks for a dense family and an MoE one. The MoE slot is deliberately
/// NOT filled by `gemma-4-26b-a4b-it-4bit`, the only MoE checkpoint on this
/// host, because Gemma is on the f16-fragile list twice over: by the `"gemma"`
/// family substring and by `text_config.final_logit_softcapping = 30.0`. The
/// policy therefore declines to convert it, both arms run identical bf16, and
/// the two PPL tables agree to four decimal places while testing nothing. That
/// is the policy behaving correctly and it is worthless as evidence.
///
/// A quantized checkpoint is substituted instead. It covers a different axis
/// than MoE, and the axis it covers is the one where this change does the most:
/// on a quantized model the packed planes stay u32 and only the bf16 side-data
/// converts, which is what moves decode by 2.40x. The MoE criterion stays
/// unmet until a non-fragile MoE checkpoint is available.
#[test]
#[ignore = "requires qwen3.8-27B-4bit weights and a CUDA device — \
            run with --release --features cuda -- --ignored test_pre_ampere_f16_ppl_quantized --nocapture"]
fn test_pre_ampere_f16_ppl_quantized() {
    run_ppl_sweep(
        "mlx-community/qwen3.8-27B-4bit",
        PPL_CHUNKS_LARGE,
        &PPL_WINDOWS_LARGE,
    );
}
