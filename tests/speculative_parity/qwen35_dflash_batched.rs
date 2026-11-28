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

//! Batched (B = 4) DFlash on a Qwen 3.5 target against four B = 1 runs of the
//! same round loop (issues #2353, #2851).
//!
//! Issue #2353: before its fix, a round whose rows accepted different draft
//! counts left the shorter rows' rejected K/V tail zeroed but attended, and
//! the batched stream drifted from the B = 1 stream a few rounds later. The
//! fix clamps every round to a uniform accept, so the rectangular cache never
//! carries a phantom key.
//!
//! Issue #2851: the hard gate above still failed after the clamp, because the
//! batched `[B, block]` target verify forward is not bitwise identical to the
//! `[1, block]` forward. The identical-prompts control isolates that from the
//! accept policy, and both tests now admit a first mismatch only inside the
//! calibrated evaluation-path jitter margin.
use super::qwen35_dflash_harness::{
    ARBITER_SEQ_BASE_CONTROL, ARBITER_SEQ_BASE_GATE, BATCHED_VERIFY_JITTER_MARGIN, TEXTS,
    first_mismatch_arbiters, mean_accepts, ragged_round, rows_outside_margin, run_burst,
};
use super::{REACHABLE_PAIRINGS, pairing_present};

/// AC-3 of #2353. A B = 4 DFlash burst on the Qwen 3.5 4B target with its
/// DFlash drafter emits, for every row, the greedy tokens a B = 1 burst of the
/// same round loop emits for that row, and at least one round had ragged
/// accepts (otherwise the test proves nothing). Two B = 1 runs of row 0 pin
/// the baseline's own self-parity first.
///
/// Byte-equal rows pass outright. A row that parts from B = 1 at one token is
/// admitted only when the one-shot B = 1 arbiter puts both tokens within
/// [`BATCHED_VERIFY_JITTER_MARGIN`] logits of its top token (issue #2851).
///
/// The reference is the batched round loop at B = 1, not `DFlashGenerator`
/// (the server's B = 1 arm), so the only variable is the batch. The served
/// concurrent-versus-sequential comparison is the `mlxcel-server` smoke run
/// recorded on #2353's PR.
#[test]
#[ignore = "real-model heavy (Qwen3.5-4B target + DFlash drafter, B=4 vs B=1 batched runs); CI hardware lane only"]
fn greedy_parity_dflash_qwen35_batched_b4_matches_b1() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};

    let pairing = &REACHABLE_PAIRINGS[0];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (batched B=4): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    super::common::apply_server_mlx_cache_defaults();
    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded, tokenizer) = load_model(&target_path).expect("target model must load");
    let model: &mlxcel::models::Qwen35Model = match &loaded {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => m,
        LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => &vlm.text_model,
        _ => panic!("DFlash pairing requires a Qwen 3.5 family target"),
    };
    let encoded: Vec<Vec<i32>> = TEXTS
        .iter()
        .map(|t| {
            tokenizer
                .encode(t, true)
                .expect("prompt must tokenize")
                .into_iter()
                .map(|id| id as i32)
                .collect()
        })
        .collect();
    let len = encoded.iter().map(Vec::len).min().expect("four prompts");
    let prompts: Vec<Vec<i32>> = encoded.into_iter().map(|p| p[..len].to_vec()).collect();

    let (mut drafter, kind) =
        load_drafter(&draft_path, Some(DrafterKind::Dflash)).expect("DFlash drafter must load");
    assert_eq!(kind, DrafterKind::Dflash);
    let block_size = pairing.block_size;

    // Baseline self-parity: the B = 1 reference must reproduce itself.
    let (first, d) = run_burst(model, drafter, &prompts[..1], block_size);
    let (again, d) = run_burst(model, d, &prompts[..1], block_size);
    drafter = d;
    assert_eq!(
        first.tokens, again.tokens,
        "B = 1 DFlash is not self-consistent on this host; the B = 4 comparison would be noise"
    );

    let mut reference = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let (run, d) = run_burst(model, drafter, std::slice::from_ref(prompt), block_size);
        drafter = d;
        let (walk, committed) = mean_accepts(&run);
        eprintln!(
            "[dflash B=1] {} tokens, mean accept walk {walk:.3} committed {committed:.3}",
            run.tokens[0].len()
        );
        reference.push(run.tokens.into_iter().next().expect("one row"));
    }

    let (batched, _drafter) = run_burst(model, drafter, &prompts, block_size);
    let (walk, committed) = mean_accepts(&batched);
    eprintln!(
        "[dflash B=4] prompt length {len}, rounds {}, clamped rounds {}, mean accept per \
         row-round: walk {walk:.3}, committed {committed:.3}",
        batched.accept_lens[0].len(),
        batched.clamped_rounds,
    );
    let ragged = ragged_round(&batched);
    // Issue #2851: byte equality is the ideal. Failing that, a first mismatch
    // is admitted only when the one-shot B = 1 arbiter ranks both tokens
    // within `BATCHED_VERIFY_JITTER_MARGIN`, which separates evaluation-path
    // kernel jitter from a structural break.
    let (ties, structural) = first_mismatch_arbiters(
        model,
        "dflash B=4",
        &prompts,
        &batched,
        &reference,
        ARBITER_SEQ_BASE_GATE,
    );
    let outside = rows_outside_margin(&ties, BATCHED_VERIFY_JITTER_MARGIN);
    assert!(
        structural.is_empty(),
        "batched DFlash cannot be arbitrated against B = 1:\n{}",
        structural.join("\n")
    );
    assert!(
        ragged,
        "no round had ragged accepts; the prompts do not exercise the clamp"
    );
    assert!(
        outside.is_empty(),
        "batched DFlash diverged from B = 1 beyond the evaluation-path jitter margin \
         of {BATCHED_VERIFY_JITTER_MARGIN} logits:\n{}",
        outside.join("\n")
    );
    eprintln!(
        "[dflash B=4] PASS: all {} rows match their B = 1 references (byte-identical or \
         within the {BATCHED_VERIFY_JITTER_MARGIN}-logit evaluation-path margin)",
        batched.tokens.len(),
    );
}

/// Issue #2851: four identical prompts in one B = 4 burst. Every row drafts
/// the same block from the same prefix, so every round accepts its full walk
/// (`clamped_rounds == 0`) and no round is ragged. The uniform clamp and the
/// accept policy are therefore inert, and any divergence from a B = 1 run of
/// the same round loop is produced by the batched target forward itself:
/// identical rows rule out a ragged accept or a stale prefix as the cause.
///
/// The control asserts `clamped_rounds == 0` and no ragged round, then applies
/// the same arbitrated parity rule the hard gate applies: each row must be
/// byte-identical or sit within [`BATCHED_VERIFY_JITTER_MARGIN`] of the
/// one-shot B = 1 arbiter at its first mismatch. Its margin-removal arm must
/// reject those same mismatches at margin 0, which is what makes the margin
/// evidence rather than a widened tolerance.
#[test]
#[ignore = "real-model heavy (Qwen3.5-4B target + DFlash drafter, four identical rows at B=4 vs B=1); CI hardware lane only"]
fn identical_prompts_dflash_qwen35_b4_matches_b1() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};

    let pairing = &REACHABLE_PAIRINGS[0];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (identical prompts, batched B=4): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    super::common::apply_server_mlx_cache_defaults();
    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded, tokenizer) = load_model(&target_path).expect("target model must load");
    let model: &mlxcel::models::Qwen35Model = match &loaded {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => m,
        LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => &vlm.text_model,
        _ => panic!("DFlash pairing requires a Qwen 3.5 family target"),
    };
    let encoded: Vec<Vec<i32>> = TEXTS
        .iter()
        .map(|t| {
            tokenizer
                .encode(t, true)
                .expect("prompt must tokenize")
                .into_iter()
                .map(|id| id as i32)
                .collect()
        })
        .collect();
    let len = encoded.iter().map(Vec::len).min().expect("four prompts");
    let prompt = encoded[0][..len].to_vec();
    let rows = vec![prompt; 4];

    let (drafter, kind) =
        load_drafter(&draft_path, Some(DrafterKind::Dflash)).expect("DFlash drafter must load");
    assert_eq!(kind, DrafterKind::Dflash);
    let block_size = pairing.block_size;

    let (reference_run, drafter) = run_burst(model, drafter, &rows[..1], block_size);
    let reference = reference_run.tokens.into_iter().next().expect("one row");
    let (batched, _drafter) = run_burst(model, drafter, &rows, block_size);
    let ragged = ragged_round(&batched);
    eprintln!(
        "[dflash identical B=4] prompt length {len}, rounds {}, clamped rounds {}, ragged {ragged}",
        batched.accept_lens[0].len(),
        batched.clamped_rounds,
    );
    assert_eq!(
        batched.clamped_rounds, 0,
        "identical rows clamped in {} round(s); the control is only discriminating \
         when every round commits its walk",
        batched.clamped_rounds,
    );
    assert!(
        !ragged,
        "identical rows produced a ragged accept round; the control is only \
         discriminating when no round clamps"
    );
    let want: Vec<Vec<i32>> = (0..batched.tokens.len())
        .map(|_| reference.clone())
        .collect();
    let (ties, structural) = first_mismatch_arbiters(
        model,
        "dflash identical B=4",
        &rows,
        &batched,
        &want,
        ARBITER_SEQ_BASE_CONTROL,
    );
    let outside = rows_outside_margin(&ties, BATCHED_VERIFY_JITTER_MARGIN);
    assert!(
        structural.is_empty(),
        "identical-row batched DFlash cannot be arbitrated against B = 1:\n{}",
        structural.join("\n")
    );
    assert!(
        outside.is_empty(),
        "batched DFlash with identical rows diverged from B = 1 beyond the \
         evaluation-path jitter margin of {BATCHED_VERIFY_JITTER_MARGIN} logits:\n{}",
        outside.join("\n")
    );
    // Margin-removal control. The margin is only evidence when it is
    // load-bearing: with it set to 0, every observed first mismatch must be
    // rejected. If the batched forward ever becomes byte-identical this arm
    // says so instead of asserting, because the margin would then no longer be
    // needed.
    if ties.is_empty() {
        eprintln!(
            "[dflash identical B=4] all rows byte-identical; the jitter margin is not \
             load-bearing on this tree"
        );
    } else {
        let at_zero = rows_outside_margin(&ties, 0.0);
        eprintln!(
            "[dflash identical B=4] margin-removal control: {} of {} first mismatches \
             rejected at margin 0",
            at_zero.len(),
            ties.len(),
        );
        assert!(
            !at_zero.is_empty(),
            "margin is not load-bearing: every first mismatch also passed at margin 0, \
             so BATCHED_VERIFY_JITTER_MARGIN admits nothing a byte-equality gate would \
             reject"
        );
    }
    eprintln!(
        "[dflash identical B=4] PASS: clamped_rounds=0, no ragged round, and all {} rows \
         match B = 1 within the {BATCHED_VERIFY_JITTER_MARGIN}-logit evaluation-path margin",
        batched.tokens.len(),
    );
}
