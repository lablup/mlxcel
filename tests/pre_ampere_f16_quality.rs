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
const PPL_CHUNKS_LARGE: usize = 16;

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
    // `MLXCEL_PPL_CORPUS` scores a different file. An instruction-tuned model on
    // raw wikitext is out of its own distribution, and a low score there is a
    // domain statement, not a defect; pointing this at the model's own greedy
    // output separates the two.
    let corpus_path = match std::env::var_os("MLXCEL_PPL_CORPUS") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("wikitext2_excerpt.txt"),
    };
    let corpus = fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("corpus missing at {corpus_path:?}: {e}"));

    // Each window is prefixed with the model's BOS when it declares one, and
    // the BOS position is then excluded from the score below. Without it a
    // window starts mid-stream in a state the model was never trained to see,
    // and families that depend on BOS land far outside their normal operating
    // range: Gemma scores 438 to 5550 unprefixed, which is not a number a 26B
    // model produces on wikitext. That matters here beyond tidiness, because
    // this file exists to find f16 exponent overflow, and overflow has to be
    // looked for in the activation distribution the model actually runs in.
    let all_ids = tokenizer.encode(&corpus, false).expect("tokenize corpus");
    let ids: Vec<i32> = all_ids.iter().map(|&id| id as i32).collect();
    let bos = tokenizer.bos_token_id().map(|id| id as i32);

    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), KVCacheMode::Fp16);
    let mut total_nll = 0.0_f64;
    let mut total_targets = 0_usize;
    let mut non_finite = 0_usize;

    let chunks = chunks_wanted.min(ids.len() / window_len);
    for chunk in 0..chunks {
        let slice = &ids[chunk * window_len..(chunk + 1) * window_len];
        // Prepend BOS and drop the corpus token it predicts, so every arm
        // scores exactly the same `window_len - 1` corpus positions whether or
        // not the family has a BOS. Otherwise adding BOS would silently change
        // the denominator and make windows incomparable across models.
        let owned;
        let window: &[i32] = match bos {
            Some(bos_id) => {
                owned = std::iter::once(bos_id)
                    .chain(slice.iter().copied())
                    .collect::<Vec<i32>>();
                &owned
            }
            None => slice,
        };
        let logprobs = generator.evaluate_loglikelihoods(model, window);
        let logprobs = if bos.is_some() {
            &logprobs[1..]
        } else {
            &logprobs[..]
        };
        if std::env::var_os("MLXCEL_PPL_PER_CHUNK").is_some() {
            let n: f64 = logprobs.iter().map(|&lp| -(lp as f64)).sum();
            eprintln!(
                "  [chunk {chunk}] ppl={:.4}",
                (n / logprobs.len() as f64).exp()
            );
        }

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
        _ if std::env::var("MLXCEL_CUDA_F16_FRAGILE").is_ok_and(|v| v != "0") => {
            "f16 (forced past the fragile list)"
        }
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

// The instruction-tuned Gemma is deliberately absent from this file, and the
// reason is the useful part of a long investigation, so it is recorded rather
// than left for the next person to repeat.
//
// `gemma-4-26b-a4b-it` scores a perplexity here between 255 and 88553 depending
// on the chunk, against Llama's 15.67 on the same corpus and the same code, and
// top-1 next-token accuracy of 20.5% against Llama's 59.1%. That looks exactly
// like a broken evaluation path, and it was chased as one through six
// hypotheses: cache reuse across chunks (the caches are reset, and chunk 0
// scores identically whether run alone or as the first of four), a missing
// sliding-window mask (the window is 1024 and the collapse is already total at
// 32), a missing final-logit softcap (it is applied), a tokenizer fault (the
// round trip is exact), a logits-shape mismatch (`[1, T, V]`, rows match the
// input), and an off-by-one (row `i` predicts token `i+1` at 20.5% against 2.3%
// for token `i`). All six are wrong.
//
// The explanation is that the instruction-tuned checkpoint cannot continue raw
// text at all. Asked to continue "The printing press was invented in Europe
// during the fifteenth century. Its most important consequence was" with
// `--no-chat-template`, it emits "ownce of consequence-wise-wise-wise-..." and
// never recovers, while Llama-3.1-8B-Instruct on the identical invocation
// continues fluently through Gutenberg and the 1440s. Inside its chat template
// it answers the same question well. Wikitext perplexity measures precisely the
// ability it no longer has, so it is an inapplicable metric rather than an
// inaccurate one, and a dtype comparison computed inside it means nothing.
//
// The base checkpoint below does not have that problem, which is why it, and
// not the instruction-tuned one, fills the MoE slot.

/// MoE family, on the base checkpoint rather than the instruction-tuned one.
///
/// `gemma-4-26b-a4b-4bit` continues raw text correctly on the same prompt where
/// its `-it` sibling collapses, so perplexity is a metric it can actually be
/// held to. It is also the checkpoint that makes the f16-fragile exemption in
/// `is_f16_fragile_below_ampere` testable: it carries the same two triggers,
/// the `gemma4` model_type and `final_logit_softcapping = 30.0`, so a
/// regression here would say the exemption was wrong.
#[test]
#[ignore = "requires gemma-4-26b-a4b-4bit weights and a CUDA device — \
            run with --release --features cuda -- --ignored test_pre_ampere_f16_ppl_moe --nocapture"]
fn test_pre_ampere_f16_ppl_moe() {
    run_ppl_sweep(
        "mlx-community/gemma-4-26b-a4b-4bit",
        PPL_CHUNKS_LARGE,
        &PPL_WINDOWS_LARGE,
    );
}

/// The families still on the f16-fragile list below sm_80, under test.
///
/// Gemma was removed from that list on evidence; these two are what is left of
/// it that this host can measure, and they are a different case. Apertus squares
/// through xIELU and gpt-oss carries a wide dynamic range, so the argument that
/// acquitted Gemma, that `tanh(x / c) * c` bounds a value rather than growing
/// it, does not transfer. Both continue raw text well enough to be scored,
/// unlike `gemma-4-26b-a4b-it`, so perplexity is a metric they can be held to.
///
/// Run with `MLXCEL_CUDA_F16_FRAGILE=1` for the f16 arm, since the list still
/// keeps them on bf16 by default, and against the plain default for the control.
///
/// Cohere is absent: `c4ai-command-r7b-12-2024-4bit` ships F16 weights with zero
/// bf16 tensors, so the exclusion is a no-op on it and measuring it here would
/// establish nothing about the families that do carry bf16.
#[test]
#[ignore = "requires Apertus-8B-Instruct-2509-4bit weights and a CUDA device — \
            run with --release --features cuda -- --ignored test_pre_ampere_f16_ppl_apertus --nocapture"]
fn test_pre_ampere_f16_ppl_apertus() {
    run_ppl_sweep(
        "mlx-community/Apertus-8B-Instruct-2509-4bit",
        PPL_CHUNKS_LARGE,
        &PPL_WINDOWS_LARGE,
    );
}

#[test]
#[ignore = "requires gpt-oss-20b-MXFP4-Q8 weights and a CUDA device — \
            run with --release --features cuda -- --ignored test_pre_ampere_f16_ppl_gpt_oss --nocapture"]
fn test_pre_ampere_f16_ppl_gpt_oss() {
    run_ppl_sweep(
        "mlx-community/gpt-oss-20b-MXFP4-Q8",
        PPL_CHUNKS_LARGE,
        &PPL_WINDOWS_LARGE,
    );
}

/// Cohere, which ships f16 weights and therefore answers the fragile question
/// without a conversion.
///
/// `c4ai-command-r7b-12-2024-4bit` carries 483 F16 tensors and zero bf16, so
/// `MLXCEL_CUDA_F16_NORMALIZE` has nothing to convert and both arms are the same
/// run. That makes a before/after comparison impossible and was first written
/// off here as unmeasurable, which was the wrong reading: the fragile list
/// claims f16 is unsafe for this family, and this checkpoint already executes in
/// f16 on every backend. If it scores normally, the claim is contradicted by the
/// shipping configuration rather than by an experiment, which is stronger
/// evidence than a conversion delta would have been.
#[test]
#[ignore = "requires c4ai-command-r7b-12-2024-4bit weights and a CUDA device — \
            run with --release --features cuda -- --ignored test_pre_ampere_f16_ppl_cohere --nocapture"]
fn test_pre_ampere_f16_ppl_cohere() {
    run_ppl_sweep(
        "mlx-community/c4ai-command-r7b-12-2024-4bit",
        PPL_CHUNKS_LARGE,
        &PPL_WINDOWS_LARGE,
    );
}

/// Second family: a large quantized checkpoint.
///
/// #1542 asks for a dense family and an MoE one. The MoE slot stays open: the
/// only MoE checkpoint on this host is Gemma, which the comment above explains
/// cannot be scored this way at all.
///
/// A quantized checkpoint covers the remaining axis, and it is the one where
/// this change does the most: on a quantized model the packed planes stay u32
/// and only the bf16 side-data converts, which is what moves decode by 2.40x.
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
