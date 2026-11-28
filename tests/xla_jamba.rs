// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0
//! Jamba (issue #3349) device gates against independent HF fp32 references:
//! the seeded MoE fixture's per-step logits on the single-sequence and ragged
//! engines, and the real checkpoint's 40-token oracle.
#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use mlxcel_xla::{IreeLlama, IreeRaggedLlama};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    prompt: Vec<i32>,
    generated: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

const VOCAB: usize = 32;

fn check_logits(got: &[f32], want: &[f32], context: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{context}");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{context}: non-finite logits"
    );
    let error = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(error < 3e-4, "{context}: max_abs={error}");
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(got), argmax(want), "{context}");
    error
}

/// `MLXCEL_JAMBA_FIXTURE` is the `--out` of `spike/openxla/jamba_moe_fixture.py`:
/// the engine loads its stacked `mlx/` layout and compares with `oracle.json`,
/// which the HF model computed from the native `hf/` checkpoint.
#[test]
#[ignore = "requires pinned IREE and the fixture from spike/openxla/jamba_moe_fixture.py"]
fn jamba_moe_hf_synthetic_parity() {
    let root = std::path::PathBuf::from(
        std::env::var("MLXCEL_JAMBA_FIXTURE").expect("MLXCEL_JAMBA_FIXTURE"),
    );
    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".into());
    let model = root.join("mlx");
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let steps = cases[0].generated.len();
    let mut worst = 0.0f32;

    let mut single = IreeLlama::load(&model, &device, 32).unwrap();
    for (index, case) in cases.iter().enumerate() {
        let mut token = single.prefill_first(&case.prompt).unwrap();
        assert_eq!(token, case.generated[0], "case {index} prefill");
        for step in 1..steps {
            token = single
                .decode(token, (case.prompt.len() + step - 1) as i32)
                .unwrap();
            assert_eq!(
                token, case.generated[step],
                "case {index} single step {step}"
            );
        }
    }

    let mut ragged = IreeRaggedLlama::load(&model, &device, 4, 32).unwrap();
    for (slot, case) in cases.iter().enumerate() {
        let got = ragged.prefill_slot_logits(slot, &case.prompt).unwrap();
        worst = worst.max(check_logits(
            &got,
            &case.logits[0],
            &format!("prefill slot {slot}"),
        ));
    }
    for step in 1..steps {
        let tokens: Vec<i32> = cases.iter().map(|c| c.generated[step - 1]).collect();
        let positions: Vec<i32> = cases
            .iter()
            .map(|c| (c.prompt.len() + step - 1) as i32)
            .collect();
        let got = ragged
            .decode_ragged_logits_with_modes(&[0; 4], &tokens, &positions, &positions)
            .unwrap();
        for (slot, case) in cases.iter().enumerate() {
            worst = worst.max(check_logits(
                &got[slot * VOCAB..(slot + 1) * VOCAB],
                &case.logits[step],
                &format!("step {step} slot {slot}"),
            ));
        }
    }
    // Re-admission replaces both the KV rows and the Mamba1 state of a used slot.
    for (slot, case) in cases.iter().rev().enumerate() {
        let got = ragged.prefill_slot_logits(slot, &case.prompt).unwrap();
        worst = worst.max(check_logits(
            &got,
            &case.logits[0],
            &format!("recycle slot {slot}"),
        ));
    }
    eprintln!(
        "Jamba MoE fixture: {} cases x {steps} steps match HF on {device}, max_abs={worst:e}",
        cases.len()
    );
}

#[test]
#[ignore = "requires the original Jamba checkpoint and its independent 40-token HF oracle"]
fn jamba_original_checkpoint_oracle_tokens() {
    #[derive(Deserialize)]
    struct Oracle {
        prompt_ids: Vec<i32>,
        ref_token_ids: Vec<i32>,
    }
    let model = std::env::var("MLXCEL_JAMBA_MODEL").expect("MLXCEL_JAMBA_MODEL");
    let oracle = std::env::var("MLXCEL_JAMBA_ORACLE").expect("MLXCEL_JAMBA_ORACLE");
    let oracle: Oracle = serde_json::from_slice(&std::fs::read(oracle).unwrap()).unwrap();
    assert!(
        oracle.ref_token_ids.len() >= 40,
        "acceptance requires at least 40 reference tokens"
    );
    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".into());
    let mut engine = IreeLlama::load(std::path::Path::new(&model), &device, 64).unwrap();
    let mut token = engine.prefill_first(&oracle.prompt_ids).unwrap();
    assert_eq!(token, oracle.ref_token_ids[0], "prefill token");
    for (step, &want) in oracle.ref_token_ids.iter().enumerate().skip(1) {
        token = engine
            .decode(token, (oracle.prompt_ids.len() + step - 1) as i32)
            .unwrap();
        assert_eq!(token, want, "decode step {step}");
    }
    eprintln!(
        "Jamba {} oracle tokens match on {device}",
        oracle.ref_token_ids.len()
    );
}
