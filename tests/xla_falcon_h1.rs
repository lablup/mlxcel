// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0
//! Independent HF fp32 synthetic gate; generation plus per-step ragged logits.
#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use mlxcel_xla::{IreeLlama, IreeRaggedLlama};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    prompt: Vec<i32>,
    generated: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn check_logits(got: &[f32], want: &[f32], context: &str) {
    assert_eq!(got.len(), want.len());
    let error = max_abs(got, want);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{context}: non-finite logits"
    );
    assert!(error < 3e-4, "{context}: max_abs={error}");
    assert_eq!(argmax(got), argmax(want), "{context}");
}

#[test]
#[ignore = "requires pinned IREE and fixtures from spike/openxla/falcon_h1_oracle.py"]
fn falcon_h1_hf_synthetic_parity() {
    let root = std::env::var("MLXCEL_FALCON_H1_FIXTURES").expect("MLXCEL_FALCON_H1_FIXTURES");
    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".into());
    for norm in ["none", "before", "after"] {
        let path = std::path::Path::new(&root).join(norm);
        let cases: Vec<Case> =
            serde_json::from_slice(&std::fs::read(path.join("oracle.json")).unwrap()).unwrap();
        let mut single = IreeLlama::load(&path, &device, 32).unwrap();
        for case in &cases {
            let mut token = single.prefill_first(&case.prompt).unwrap();
            assert_eq!(token, case.generated[0], "{norm} prefill");
            for step in 1..case.generated.len() {
                token = single
                    .decode(token, (case.prompt.len() + step - 1) as i32)
                    .unwrap();
                assert_eq!(token, case.generated[step], "{norm} single step {step}");
            }
        }
        let mut ragged = IreeRaggedLlama::load(&path, &device, 4, 32).unwrap();
        for (slot, case) in cases.iter().enumerate() {
            let got = ragged.prefill_slot_logits(slot, &case.prompt).unwrap();
            check_logits(
                &got,
                &case.logits[0],
                &format!("{norm} prefill slot {slot}"),
            );
        }
        for step in 1..8 {
            let tokens: Vec<i32> = cases.iter().map(|c| c.generated[step - 1]).collect();
            let positions: Vec<i32> = cases
                .iter()
                .map(|c| (c.prompt.len() + step - 1) as i32)
                .collect();
            let got = ragged
                .decode_ragged_logits_with_modes(&[0; 4], &tokens, &positions, &positions)
                .unwrap();
            for (slot, case) in cases.iter().enumerate() {
                check_logits(
                    &got[slot * 32..(slot + 1) * 32],
                    &case.logits[step],
                    &format!("{norm} step {step} slot {slot}"),
                );
            }
        }
        // Re-admission replaces both KV and recurrent state in an already used slot.
        for (slot, case) in cases.iter().rev().enumerate() {
            let got = ragged.prefill_slot_logits(slot, &case.prompt).unwrap();
            check_logits(
                &got,
                &case.logits[0],
                &format!("{norm} recycle slot {slot}"),
            );
        }
    }
}

#[test]
#[ignore = "requires original checkpoint and independent 40-token HF oracle"]
fn falcon_h1_original_checkpoint_oracle_tokens() {
    #[derive(Deserialize)]
    struct Oracle {
        prompt_ids: Vec<i32>,
        ref_token_ids: Vec<i32>,
    }
    let model = std::env::var("MLXCEL_FALCON_H1_MODEL").expect("MLXCEL_FALCON_H1_MODEL");
    let oracle = std::env::var("MLXCEL_FALCON_H1_ORACLE").expect("MLXCEL_FALCON_H1_ORACLE");
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
        "Falcon-H1 {} oracle tokens match on {device}",
        oracle.ref_token_ids.len()
    );
}

/// Long-position logits on the original checkpoint. Wrong RoPE scaling changes
/// only low-frequency bands, so its error grows with position and the 40-token
/// oracle cannot see it. Four prefixes of one seeded stream (32 to 3584 tokens)
/// share a ragged batch, then decode teacher-forced steps together.
#[test]
#[ignore = "requires the original checkpoint and long_oracle.json from falcon_h1_oracle.py"]
fn falcon_h1_original_checkpoint_long_positions() {
    const TOLERANCE: f32 = 1e-3;
    let model = std::env::var("MLXCEL_FALCON_H1_MODEL").expect("MLXCEL_FALCON_H1_MODEL");
    let root = std::env::var("MLXCEL_FALCON_H1_FIXTURES").expect("MLXCEL_FALCON_H1_FIXTURES");
    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".into());
    let path = std::path::Path::new(&root).join("long_oracle.json");
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(cases.len(), 4, "one case per ragged slot");
    let longest = cases.iter().map(|c| c.prompt.len()).max().unwrap();
    let steps = cases[0].generated.len();
    let capacity = (longest + steps).next_multiple_of(512);
    let mut ragged =
        IreeRaggedLlama::load(std::path::Path::new(&model), &device, 4, capacity).unwrap();
    let vocab = ragged.vocab();
    let mut failures = Vec::new();
    let mut record = |case: &Case, step: usize, got: &[f32]| {
        assert_eq!(got.len(), case.logits[step].len());
        assert!(got.iter().all(|x| x.is_finite()), "non-finite logits");
        let want = &case.logits[step];
        let error = max_abs(got, want);
        let same = argmax(got) == argmax(want);
        let position = case.prompt.len() + step - 1;
        eprintln!(
            "prefix {:>4} step {step} position {position:>4}: max_abs={error:.6e} argmax_match={same}",
            case.prompt.len()
        );
        if error >= TOLERANCE || !same {
            failures.push(format!("prefix {} step {step}", case.prompt.len()));
        }
    };
    for (slot, case) in cases.iter().enumerate() {
        let got = ragged.prefill_slot_logits(slot, &case.prompt).unwrap();
        record(case, 0, &got);
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
            record(case, step, &got[slot * vocab..(slot + 1) * vocab]);
        }
    }
    assert!(
        failures.is_empty(),
        "Falcon-H1 long-position logits diverge on {device} (tolerance {TOLERANCE}): {failures:?}"
    );
    eprintln!("Falcon-H1 long-position logits match on {device} up to position {longest}");
}
