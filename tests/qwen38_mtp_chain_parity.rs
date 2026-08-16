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

//! Qwen 3.5-family MTP **chain** parity: multi-token verify blocks vs the
//! single-token decode chain.
//!
//! ## What this pins (and what the sibling test cannot see)
//!
//! `tests/qwen35_mtp_verify_parity.rs` compares a verify block's per-position
//! argmax against a fresh full-prefix re-prefill per position — both arms are
//! multi-token block forwards, so their gated-delta recurrent state evolves
//! through the same in-block float32 carry. The MTP/DFlash **round loop**,
//! however, interleaves multi-token verify blocks into a chain whose
//! byte-parity reference is the classic **single-token** decode chain, where
//! the recurrent state is materialized (and dtype-rounded) after every
//! token. This test drives both chain shapes over the same token sequence
//! from the same prefill and reports the first argmax divergence, which is
//! the exact mechanism behind a temperature-0 MTP output drifting from
//! classic decode at a near-tie logit.
//!
//! ## Method
//!
//! 1. **Reference chain** — prefill a fixed prompt, then decode N tokens one
//!    at a time through `forward_speculative` (`T = 1` per call; the capture
//!    path's single-token branch is op-identical to classic decode).
//! 2. **Block chain** — fresh caches, same prefill, then replay the
//!    reference chain's tokens in `T = K` verify blocks (fully-accepted
//!    rounds: no rollback involved) and compare every position's argmax
//!    against the reference chain.
//!
//! A mismatch here means the multi-token forward itself (gated-delta block
//! scan and/or attention) is not byte-equal to the single-token chain — no
//! drafter, no rollback, no sampling in sight.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test qwen38_mtp_chain_parity --release --features metal,accelerate -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated as real-model heavy. Uses the first present candidate
//! checkpoint (Qwen3.8-27B preferred, smaller Qwen 3.5 sizes accepted).

mod common;

use common::repo_model_dir;

use mlxcel::models::Qwen35Model;
use mlxcel::models::qwen3_next::Qwen3NextCache;
use mlxcel::{LoadedModel, initialize_runtime, load_model};

const CANDIDATE_TARGETS: &[&str] = &[
    "qwen3.8-27b-4bit",
    "qwen3.5-0.8b-4bit",
    "qwen3.5-2b-4bit",
    "qwen3.5-4b-4bit",
];

/// Fixed in-vocab prompt ids (arbitrary; only chain agreement matters).
const PROMPT_TOKENS: &[i32] = &[785, 3974, 13876, 8533, 4290, 374, 264, 1273];

/// Chain length. Long enough to catch the observed divergence density
/// (flips showed up within 40-200 generated tokens at 27B).
const CHAIN_LEN: usize = 240;

/// Verify-block width for the block chain (draft 2 + bonus, the published
/// `block_size: 3` shape).
const BLOCK: usize = 3;

fn argmax_positions(logits: &mlxcel_core::MlxArray) -> Vec<i32> {
    let shape = mlxcel_core::array_shape(logits);
    let positions = shape[1];
    let argmax = mlxcel_core::argmax_last_axis(logits);
    mlxcel_core::eval(&argmax);
    let mut out = Vec::with_capacity(positions as usize);
    for i in 0..positions {
        let cell = mlxcel_core::slice(&argmax, &[0, i], &[1, i + 1]);
        out.push(mlxcel_core::item_i32(&cell));
    }
    out
}

/// Resolve the inner text model, reaching through the VLM wrapper (most
/// mlx-community Qwen 3.5-family checkpoints load as the VLM variant).
fn as_qwen35(loaded: &LoadedModel) -> Option<&Qwen35Model> {
    match loaded {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => Some(m),
        LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => Some(&vlm.text_model),
        _ => None,
    }
}

fn forward_tokens(model: &Qwen35Model, tokens: &[i32], caches: &mut [Qwen3NextCache]) -> Vec<i32> {
    let arr = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let out = model.forward_speculative(&arr, caches, &[]);
    argmax_positions(&out.logits)
}

#[test]
#[ignore]
fn block_verify_chain_matches_single_token_chain() {
    let target_dir = CANDIDATE_TARGETS
        .iter()
        .map(|name| repo_model_dir(name))
        .find(|dir| dir.join("config.json").exists());
    let Some(target_dir) = target_dir else {
        eprintln!("skipping: no candidate Qwen 3.5-family checkpoint under models/");
        return;
    };

    let _runtime = initialize_runtime();
    let (loaded, _tokenizer) = load_model(&target_dir).expect("load target");
    let model = as_qwen35(&loaded).expect("expected a Qwen 3.5-family target");

    // Reference chain: prefill + T=1 decode steps.
    let mut ref_caches = model.make_speculative_caches_for_test();
    let prefill_argmax = forward_tokens(model, PROMPT_TOKENS, &mut ref_caches);
    let mut chain: Vec<i32> = vec![*prefill_argmax.last().expect("prefill argmax")];
    for _ in 0..CHAIN_LEN {
        let last = *chain.last().expect("non-empty chain");
        let next = forward_tokens(model, &[last], &mut ref_caches);
        chain.push(next[0]);
    }

    // Block chain: fresh caches, same prefill, replay the reference tokens
    // in fully-accepted T=BLOCK verify rounds.
    let mut blk_caches = model.make_speculative_caches_for_test();
    let _ = forward_tokens(model, PROMPT_TOKENS, &mut blk_caches);
    let mut mismatches: Vec<(usize, i32, i32)> = Vec::new();
    let mut pos = 0usize; // index into `chain`: the block starts at chain[pos]
    while pos + BLOCK < chain.len() {
        let block: Vec<i32> = chain[pos..pos + BLOCK].to_vec();
        let argmax = forward_tokens(model, &block, &mut blk_caches);
        for (offset, &got) in argmax.iter().enumerate() {
            let expect = chain[pos + offset + 1];
            if got != expect {
                mismatches.push((pos + offset + 1, expect, got));
            }
        }
        pos += BLOCK;
    }

    if mismatches.is_empty() {
        eprintln!(
            "block chain matches the single-token chain over {} tokens (block={})",
            chain.len(),
            BLOCK
        );
    } else {
        let first = mismatches[0];
        panic!(
            "block-vs-single-token chain diverged at {} position(s) over {} tokens \
             (block={}); first at chain index {} (single-token argmax {}, block argmax {})",
            mismatches.len(),
            chain.len(),
            BLOCK,
            first.0,
            first.1,
            first.2,
        );
    }
}

// ===========================================================================
// Gated-delta chain-parity kernel: bitwise pin + isolated cost measurement.
// ===========================================================================

use mlxcel::models::gated_delta::{gated_delta_update, gated_delta_update_chain_parity};

/// Qwen3.8-27B GDN geometry: Hk=16, Hv=48, Dk=Dv=128 (f16 activations).
const GDN_HK: i32 = 16;
const GDN_HV: i32 = 48;
const GDN_D: i32 = 128;

fn gdn_inputs(
    t: i32,
    seed: f32,
) -> (
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
) {
    let fill = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * scale)
            .collect()
    };
    let qk_n = (t * GDN_HK * GDN_D) as usize;
    let v_n = (t * GDN_HV * GDN_D) as usize;
    let ab_n = (t * GDN_HV) as usize;
    let f16 = |data: &[f32], shape: &[i32]| {
        let arr = mlxcel_core::from_slice_f32(data, shape);
        mlxcel_core::astype(&arr, mlxcel_core::dtype::FLOAT16)
    };
    (
        f16(&fill(qk_n, 0.5), &[1, t, GDN_HK, GDN_D]),
        f16(&fill(qk_n, 0.4), &[1, t, GDN_HK, GDN_D]),
        f16(&fill(v_n, 0.3), &[1, t, GDN_HV, GDN_D]),
        f16(&fill(ab_n, 0.2), &[1, t, GDN_HV]),
        f16(&fill(ab_n, 0.1), &[1, t, GDN_HV]),
    )
}

fn gdn_slice_t(
    arr: &mlxcel_core::MlxArray,
    t: i32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let shape = mlxcel_core::array_shape(arr);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[1] = t;
    stops[1] = t + 1;
    mlxcel_core::slice(arr, &starts, &stops)
}

fn raw_bytes(arr: &mlxcel_core::MlxArray) -> Vec<u8> {
    mlxcel_core::eval(arr);
    mlxcel_core::array_to_raw_bytes(arr)
}

/// The chain-parity contract itself, pinned bitwise on synthetic 27B-geometry
/// inputs: a `T = 3` parity-kernel block must produce the SAME per-position
/// outputs and the SAME final state as three consecutive `T = 1` calls of the
/// standard kernel (the classic decode chain). This is the invariant the MTP
/// temperature-0 exactness gate rests on; the standard kernel fails it by
/// design (float32 in-block state carry).
#[test]
fn parity_kernel_block_is_bitwise_equal_to_single_token_chain() {
    if !mlxcel_core::metal_is_available() {
        eprintln!("skipping: chain-parity kernel is Metal-only");
        return;
    }
    let _runtime = initialize_runtime();
    let t = 3;
    let (q, k, v, a, b) = gdn_inputs(t, 0.123);
    let a_log_v: Vec<f32> = (0..GDN_HV).map(|i| (i as f32 * 0.05).cos() * 0.5).collect();
    let dt_v: Vec<f32> = (0..GDN_HV).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
    let a_log = mlxcel_core::from_slice_f32(&a_log_v, &[GDN_HV]);
    let dt_bias = mlxcel_core::from_slice_f32(&dt_v, &[GDN_HV]);

    // Block: parity kernel over T=3, state=None.
    let (y_block, state_block) =
        gated_delta_update_chain_parity(&q, &k, &v, &a, &b, &a_log, &dt_bias, None, None);

    // Chain: three standard T=1 calls, state carried between calls exactly
    // as the classic decode chain carries it (storage-dtype round trip).
    let mut state: Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> = None;
    let mut y_chain: Vec<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> = Vec::new();
    for step in 0..t {
        let (y_t, next_state) = gated_delta_update(
            &gdn_slice_t(&q, step),
            &gdn_slice_t(&k, step),
            &gdn_slice_t(&v, step),
            &gdn_slice_t(&a, step),
            &gdn_slice_t(&b, step),
            &a_log,
            &dt_bias,
            state.as_deref(),
            None,
        );
        y_chain.push(y_t);
        state = Some(next_state);
    }

    for step in 0..t {
        assert_eq!(
            raw_bytes(&gdn_slice_t(&y_block, step)),
            raw_bytes(y_chain[step as usize].as_ref().unwrap()),
            "parity block output at step {step} must be bitwise equal to the T=1 chain"
        );
    }
    assert_eq!(
        raw_bytes(&state_block),
        raw_bytes(state.unwrap().as_ref().unwrap()),
        "parity block final state must be bitwise equal to the T=1 chain state"
    );
}

/// Isolated cost of the parity kernel vs the standard kernel on the real
/// verify shapes (medians over repeated dispatches). Run manually:
///
/// ```bash
/// cargo test --test qwen38_mtp_chain_parity --release --features metal,accelerate -- --ignored --nocapture parity_kernel_cost
/// ```
#[test]
#[ignore]
fn parity_kernel_cost_vs_standard() {
    if !mlxcel_core::metal_is_available() {
        eprintln!("skipping: chain-parity kernel is Metal-only");
        return;
    }
    let _runtime = initialize_runtime();
    let a_log_v: Vec<f32> = (0..GDN_HV).map(|i| (i as f32 * 0.05).cos() * 0.5).collect();
    let dt_v: Vec<f32> = (0..GDN_HV).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
    let a_log = mlxcel_core::from_slice_f32(&a_log_v, &[GDN_HV]);
    let dt_bias = mlxcel_core::from_slice_f32(&dt_v, &[GDN_HV]);

    for t in [1_i32, 3, 4, 64] {
        let (q, k, v, a, b) = gdn_inputs(t, 0.5);
        let mut time = |parity: bool| -> f64 {
            // Warmup (includes the one-time kernel JIT).
            for _ in 0..20 {
                let (y, s) = if parity {
                    gated_delta_update_chain_parity(
                        &q, &k, &v, &a, &b, &a_log, &dt_bias, None, None,
                    )
                } else {
                    gated_delta_update(&q, &k, &v, &a, &b, &a_log, &dt_bias, None, None)
                };
                mlxcel_core::eval(&y);
                mlxcel_core::eval(&s);
            }
            let mut samples: Vec<f64> = Vec::with_capacity(100);
            for _ in 0..100 {
                let start = std::time::Instant::now();
                let (y, s) = if parity {
                    gated_delta_update_chain_parity(
                        &q, &k, &v, &a, &b, &a_log, &dt_bias, None, None,
                    )
                } else {
                    gated_delta_update(&q, &k, &v, &a, &b, &a_log, &dt_bias, None, None)
                };
                mlxcel_core::eval(&y);
                mlxcel_core::eval(&s);
                samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        };
        let std_us = time(false);
        let par_us = time(true);
        eprintln!(
            "T={t}: standard {std_us:.1} us, parity {par_us:.1} us, ratio {:.3}",
            par_us / std_us
        );
    }
}

// ===========================================================================
// Verify-forward cost scaling: where does the T>1 block spend its time?
// ===========================================================================

/// Times three shapes of the target forward per T, medians over repeats:
///
/// - `capture`: `forward_speculative` (per-position verify attention + GDN
///   snapshot capture) — the MTP/DFlash verify pass.
/// - `batched`: `forward_prefill_with_last_hidden` (batched causal
///   attention, normal GDN, no capture) — the classic prefill shape over the
///   same block.
/// - The `T = 1` `capture` row is the classic decode step cost reference.
///
/// The gap between `capture` and `batched` at the same T attributes the
/// verify machinery's own cost (per-position attention dispatches + snapshot
/// copies); the gap between `batched` at T and `T = 1` attributes how well
/// the block amortizes the weight read. Run manually:
///
/// ```bash
/// cargo test --test qwen38_mtp_chain_parity --release --features metal,accelerate -- --ignored --nocapture verify_forward_cost
/// ```
#[test]
#[ignore]
fn verify_forward_cost_scaling() {
    let Some(target_dir) = CANDIDATE_TARGETS
        .iter()
        .map(|name| repo_model_dir(name))
        .find(|dir| dir.join("config.json").exists())
    else {
        eprintln!("skipping: no candidate Qwen 3.5-family checkpoint under models/");
        return;
    };
    let _runtime = initialize_runtime();
    let (loaded, _tokenizer) = load_model(&target_dir).expect("load target");
    let model = as_qwen35(&loaded).expect("expected a Qwen 3.5-family target");
    let last_layer = [model.num_layers() - 1];

    // Warm caches with a small prompt so every timed call runs in the
    // decode-tail regime (non-empty prefix), then time repeated blocks.
    for t in [1_i32, 2, 3, 4, 8] {
        let tokens: Vec<i32> = (0..t).map(|i| 700 + i).collect();
        let arr = mlxcel_core::from_slice_i32(&tokens, &[1, t]);

        let mut time_shape = |capture: bool| -> f64 {
            let mut caches = model.make_speculative_caches_for_test();
            let prompt =
                mlxcel_core::from_slice_i32(PROMPT_TOKENS, &[1, PROMPT_TOKENS.len() as i32]);
            let (logits, _hidden) = model.forward_prefill_with_last_hidden(&prompt, &mut caches);
            mlxcel_core::eval(&logits);
            let mut samples: Vec<f64> = Vec::new();
            for i in 0..24 {
                let start = std::time::Instant::now();
                if capture {
                    let out = model.forward_speculative(&arr, &mut caches, &last_layer);
                    mlxcel_core::eval(&out.logits);
                } else {
                    let (logits, _h) = model.forward_prefill_with_last_hidden(&arr, &mut caches);
                    mlxcel_core::eval(&logits);
                }
                if i >= 4 {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        };

        let capture_ms = time_shape(true);
        let batched_ms = time_shape(false);
        eprintln!("T={t}: capture {capture_ms:.1} ms, batched {batched_ms:.1} ms");
    }
}

// ===========================================================================
// Adapter-level chain parity: the REAL MtpTarget adapter vs classic decode.
// ===========================================================================

use mlxcel::models::qwen3_5_mtp_target::Qwen35MtpTargetAdapter;
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::speculative::mtp::target::MtpTarget;

/// Classic-path chain: prefill via `LanguageModel::forward` (the exact
/// numeric path the drafter-less CLI decode takes for a text-only request on
/// this family) then `T = 1` decode steps, greedy argmax throughout.
fn classic_chain(model: &Qwen35Model, prompt: &[i32], len: usize) -> Vec<i32> {
    <Qwen35Model as LanguageModel>::reset_runtime_state(model);
    let prompt_arr = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = <Qwen35Model as LanguageModel>::forward(model, &prompt_arr, &mut [], None);
    let mut chain = vec![*argmax_positions(&logits).last().expect("prefill argmax")];
    for _ in 0..len {
        let last = *chain.last().expect("non-empty");
        let arr = mlxcel_core::from_slice_i32(&[last], &[1, 1]);
        let logits = <Qwen35Model as LanguageModel>::forward(model, &arr, &mut [], None);
        chain.push(argmax_positions(&logits)[0]);
    }
    chain
}

fn greedy() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    }
}

/// Drive the real [`Qwen35MtpTargetAdapter`] with **perfect drafts** (the
/// classic chain's own tokens): every round fully accepts, no rollback ever
/// runs. Any divergence from the classic chain is therefore in the adapter's
/// prefill or the multi-token verify forward itself.
#[test]
#[ignore]
fn adapter_chain_with_perfect_drafts_matches_classic_chain() {
    let Some(target_dir) = CANDIDATE_TARGETS
        .iter()
        .map(|name| repo_model_dir(name))
        .find(|dir| dir.join("config.json").exists())
    else {
        eprintln!("skipping: no candidate Qwen 3.5-family checkpoint under models/");
        return;
    };
    let _runtime = initialize_runtime();
    let (loaded, _tokenizer) = load_model(&target_dir).expect("load target");
    let model = as_qwen35(&loaded).expect("expected a Qwen 3.5-family target");

    let chain = classic_chain(model, PROMPT_TOKENS, 240);

    <Qwen35Model as LanguageModel>::reset_runtime_state(model);
    let adapter = Qwen35MtpTargetAdapter::new(model, None);
    let sampler = greedy();
    let logprobs = LogprobsConfig::default();
    let (bonus, _seed, _lp) = adapter.prefill_and_seed(PROMPT_TOKENS, &sampler, &[], &logprobs);
    assert_eq!(
        bonus, chain[0],
        "first bonus from the adapter prefill differs from the classic prefill argmax"
    );

    let mut mismatches: Vec<(usize, i32, i32)> = Vec::new();
    let mut pos = 0usize;
    while pos + BLOCK < chain.len() {
        let verify_input: Vec<i32> = chain[pos..pos + BLOCK].to_vec();
        let out = adapter.verify_forward(&verify_input, &sampler, &logprobs);
        for (offset, &got) in out.target_tokens.iter().enumerate() {
            let expect = chain[pos + offset + 1];
            if got != expect {
                mismatches.push((pos + offset + 1, expect, got));
            }
        }
        // Fully accepted: BLOCK - 1 drafts all matched (by construction the
        // drafts ARE the classic tokens; a target mismatch is recorded above
        // but the chain replay continues on the classic tokens).
        let _ = adapter.verify_finalize(BLOCK - 1, BLOCK, out.captured);
        pos += BLOCK;
    }

    assert!(
        mismatches.is_empty(),
        "adapter verify chain (perfect drafts, no rollback) diverged from classic decode at \
         {} position(s); first at chain index {} (classic {}, verify {})",
        mismatches.len(),
        mismatches[0].0,
        mismatches[0].1,
        mismatches[0].2,
    );
    eprintln!(
        "adapter perfect-draft chain matches classic decode over {} tokens",
        chain.len()
    );
}

/// Drive the adapter with **always-wrong drafts**: every round rejects both
/// drafts (`accepted = 0`), so `verify_finalize` runs the KV trim + GDN
/// rollback replay on every round. Position 0 of each verify block only
/// conditions on the accepted prefix, so its argmax must still follow the
/// classic chain; any divergence isolates the rollback path.
#[test]
#[ignore]
fn adapter_chain_with_rejected_drafts_matches_classic_chain() {
    let Some(target_dir) = CANDIDATE_TARGETS
        .iter()
        .map(|name| repo_model_dir(name))
        .find(|dir| dir.join("config.json").exists())
    else {
        eprintln!("skipping: no candidate Qwen 3.5-family checkpoint under models/");
        return;
    };
    let _runtime = initialize_runtime();
    let (loaded, _tokenizer) = load_model(&target_dir).expect("load target");
    let model = as_qwen35(&loaded).expect("expected a Qwen 3.5-family target");

    let chain = classic_chain(model, PROMPT_TOKENS, 160);

    <Qwen35Model as LanguageModel>::reset_runtime_state(model);
    let adapter = Qwen35MtpTargetAdapter::new(model, None);
    let sampler = greedy();
    let logprobs = LogprobsConfig::default();
    let (bonus, _seed, _lp) = adapter.prefill_and_seed(PROMPT_TOKENS, &sampler, &[], &logprobs);
    assert_eq!(bonus, chain[0], "first bonus differs from classic prefill");

    let mut mismatches: Vec<(usize, i32, i32)> = Vec::new();
    for pos in 0..chain.len() - 1 {
        // Drafts guaranteed wrong: shift the classic continuation by one id.
        let wrong = |tok: i32| (tok + 1) % 1000 + 10;
        let verify_input = vec![chain[pos], wrong(chain[pos + 1]), wrong(chain[pos + 1])];
        let out = adapter.verify_forward(&verify_input, &sampler, &logprobs);
        let got = out.target_tokens[0];
        let expect = chain[pos + 1];
        if got != expect {
            mismatches.push((pos + 1, expect, got));
        }
        // Zero accepted: rollback replay runs every round.
        let _ = adapter.verify_finalize(0, BLOCK, out.captured);
    }

    assert!(
        mismatches.is_empty(),
        "adapter verify chain (rejected drafts, rollback every round) diverged from classic \
         decode at {} position(s); first at chain index {} (classic {}, verify {})",
        mismatches.len(),
        mismatches[0].0,
        mismatches[0].1,
        mismatches[0].2,
    );
    eprintln!(
        "adapter rejected-draft chain matches classic decode over {} tokens",
        chain.len()
    );
}
