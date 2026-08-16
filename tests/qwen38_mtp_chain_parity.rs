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
/// (first flips showed up within 40-80 generated tokens at 27B).
const CHAIN_LEN: usize = 96;

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

fn forward_tokens(
    model: &Qwen35Model,
    tokens: &[i32],
    caches: &mut [Qwen3NextCache],
) -> Vec<i32> {
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
