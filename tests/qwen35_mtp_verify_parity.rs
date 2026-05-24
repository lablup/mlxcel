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

//! Qwen 3.5 MTP target-verify logit / sampling parity.
//!
//! ## What this pins
//!
//! The MTP / DFlash round loop verifies a multi-token draft block in a
//! single `Qwen35Model::forward_speculative` pass and takes the target's
//! per-position argmax to decide which proposals to accept. For the
//! speculative output to be byte-identical to the drafter-less greedy
//! decode at `temperature = 0`, every verify position's logits must agree
//! with what single-token decode would have produced at the same absolute
//! position. Upstream mlx-vlm PR #1188 fixed a drift here for Qwen 3.5 by
//! computing verify-mode attention per query position; this test is the
//! direct regression gate for that fix.
//!
//! ## Method (no drafter needed)
//!
//! Pick a fixed prefix sequence. Then:
//!
//! 1. **Verify forward** — run the whole `[1, L]` sequence through
//!    `forward_speculative` (the multi-token verify path, which routes
//!    full-attention layers through the per-query-position causal
//!    attention) and read the per-position argmax `verify_argmax[i]`.
//! 2. **Decode reference** — for each position `i`, run a *fresh* prefill
//!    of the prefix `tokens[..=i]` through `forward_speculative` with a
//!    single-token tail so the final position takes the standard
//!    single-query decode SDPA branch, and read its last-position argmax
//!    `decode_argmax[i]`.
//! 3. Assert `verify_argmax[i] == decode_argmax[i]` for every position.
//!
//! Both paths share the same projections, RoPE, gated-delta recurrence and
//! LM head; the only difference is the attention shape over the block. So
//! any mismatch is exactly the verification-drift bug this issue targets.
//! Using `forward_speculative` for both the block forward and the
//! per-position reference keeps the comparison apples-to-apples (same
//! capture path, same caches) while still exercising the two distinct
//! attention branches.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test qwen35_mtp_verify_parity --release -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated as real-model heavy: it loads a Qwen 3.5 4-bit
//! checkpoint. The CI hardware lane runs it on a fixed cadence; dev runs
//! skip it by default.

mod common;

use common::repo_model_dir;

use mlxcel::models::Qwen35Model;
use mlxcel::models::qwen3_next::Qwen3NextCache;
use mlxcel::{LoadedModel, initialize_runtime, load_model};

/// Candidate Qwen 3.5 text checkpoints, smallest first. The test uses the
/// first one present on disk so it runs on hosts that only fetched one
/// size.
const CANDIDATE_TARGETS: &[&str] = &["qwen3.5-0.8b-4bit", "qwen3.5-2b-4bit", "qwen3.5-4b-4bit"];

/// Fixed prefix token ids fed through both paths. Arbitrary in-vocab ids;
/// the exact tokens do not matter — only that the verify and decode paths
/// agree on the argmax at every position. Length is the verify-block span.
const PREFIX_TOKENS: &[i32] = &[785, 3974, 13876, 8533, 4290, 374, 264, 1273, 315];

/// Read the per-position argmax token ids out of a `[1, L, vocab]` logits
/// tensor.
fn per_position_argmax(logits: &mlxcel_core::MlxArray) -> Vec<i32> {
    let shape = mlxcel_core::array_shape(logits);
    assert_eq!(shape.len(), 3, "expected [1, L, vocab] logits");
    let l = shape[1];
    let argmax = mlxcel_core::argmax_last_axis(logits); // [1, L]
    mlxcel_core::eval(&argmax);
    let mut out = Vec::with_capacity(l as usize);
    for i in 0..l {
        // argmax is [1, L]; slice cell (0, i).
        let cell = mlxcel_core::slice(&argmax, &[0, i], &[1, i + 1]);
        out.push(mlxcel_core::item_i32(&cell));
    }
    out
}

/// Resolve the inner text `Qwen35Model` from a loaded checkpoint, or `None`
/// for non-Qwen-3.5 variants.
///
/// `mlx-community` publishes most Qwen 3.5 checkpoints — including the
/// text-only sizes — as `Qwen3_5ForConditionalGeneration`, so they load as
/// the VLM-wrapped variant. The wrapper's `text_model` is the `Qwen35Model`
/// the verify pass runs on, so we reach into it for the VLM variants too
/// (mirrors `tests/speculative_parity.rs`).
fn as_qwen35(loaded: &LoadedModel) -> Option<&Qwen35Model> {
    match loaded {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => Some(m),
        LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => Some(&vlm.text_model),
        _ => None,
    }
}

/// Greedy argmax at the final position of `prefix`, computed via a fresh
/// single-token-tail decode so the last position takes the standard
/// single-query decode attention branch (`l == 1`).
fn decode_argmax_at_last(model: &Qwen35Model, prefix: &[i32]) -> i32 {
    assert!(!prefix.is_empty());
    let mut caches: Vec<Qwen3NextCache> = model.make_speculative_caches_for_test();

    // Prefill everything except the last token (if any), then forward the
    // last token alone so its forward uses the single-query decode SDPA.
    let split = prefix.len() - 1;
    if split > 0 {
        let head = &prefix[..split];
        let head_arr = mlxcel_core::from_slice_i32(head, &[1, head.len() as i32]);
        let out = model.forward_speculative(&head_arr, &mut caches, &[]);
        mlxcel_core::eval(&out.logits);
    }
    let tail = &prefix[split..]; // exactly one token
    let tail_arr = mlxcel_core::from_slice_i32(tail, &[1, tail.len() as i32]);
    let out = model.forward_speculative(&tail_arr, &mut caches, &[]);
    let argmax = per_position_argmax(&out.logits);
    *argmax
        .last()
        .expect("decode produced at least one logit row")
}

/// The Qwen 3.5 MTP verify pass must produce per-position
/// argmax tokens identical to single-token decode at the same positions.
///
/// This is the load-bearing correctness gate for the verification-drift /
/// sampling-parity fix ported from upstream mlx-vlm PR #1188.
#[test]
#[ignore = "real-model heavy (loads a Qwen 3.5 4-bit checkpoint); runs in CI hardware lane only"]
fn qwen35_verify_block_argmax_matches_single_token_decode() {
    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    // Resolve the first available target checkpoint.
    let target = CANDIDATE_TARGETS
        .iter()
        .map(|name| (name, repo_model_dir(name)))
        .find(|(_, path)| path.exists());
    let (name, target_path) = match target {
        Some((name, path)) => (name, path),
        None => {
            eprintln!(
                "Skipping qwen35_verify_block_argmax_matches_single_token_decode: \
                 no Qwen 3.5 checkpoint on disk (looked for {CANDIDATE_TARGETS:?}). \
                 Run `mlxcel download mlx-community/Qwen3.5-0.8B-4bit` to populate.",
            );
            return;
        }
    };
    eprintln!("[mtp-parity] using target {name} at {target_path:?}");

    let (loaded, _tokenizer) = load_model(&target_path).expect("Qwen 3.5 target must load");
    let model = as_qwen35(&loaded).unwrap_or_else(|| {
        panic!(
            "checkpoint {name} did not load as a Qwen 3.5 text variant \
             (Qwen35 / Qwen35Moe); got a different LoadedModel variant"
        )
    });

    // ---- Verify forward: the whole prefix in one multi-token pass. ----
    let prefix_arr = mlxcel_core::from_slice_i32(PREFIX_TOKENS, &[1, PREFIX_TOKENS.len() as i32]);
    let mut verify_caches: Vec<Qwen3NextCache> = model.make_speculative_caches_for_test();
    let verify_out = model.forward_speculative(&prefix_arr, &mut verify_caches, &[]);
    let verify_argmax = per_position_argmax(&verify_out.logits);
    assert_eq!(
        verify_argmax.len(),
        PREFIX_TOKENS.len(),
        "verify forward must emit one logit row per input position"
    );

    // ---- Decode reference: per-position single-token decode. ----
    //
    // Compare every position from index 1 onward (position 0 has no prefix
    // to drift, so it is trivially equal; positions >= 1 are where the
    // batched-vs-per-position attention difference can surface).
    let mut mismatches: Vec<(usize, i32, i32)> = Vec::new();
    for i in 1..PREFIX_TOKENS.len() {
        let decode_tok = decode_argmax_at_last(model, &PREFIX_TOKENS[..=i]);
        if verify_argmax[i] != decode_tok {
            mismatches.push((i, verify_argmax[i], decode_tok));
        }
    }

    assert!(
        mismatches.is_empty(),
        "[mtp-parity] Qwen 3.5 verify-pass argmax diverged from single-token decode at {} of {} \
         positions (verify != decode): {:?}. This is the verification-drift / sampling-parity \
         bug upstream mlx-vlm PR #1188 fixed — the multi-token verify forward must compute \
         attention per query position so its logits match single-token decode.",
        mismatches.len(),
        PREFIX_TOKENS.len() - 1,
        mismatches,
    );

    eprintln!(
        "[mtp-parity] PASS: all {} verify positions match single-token decode argmax",
        PREFIX_TOKENS.len() - 1,
    );

    drop(loaded);
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();
}
