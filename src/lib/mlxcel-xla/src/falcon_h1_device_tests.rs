// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0
//! Independent HF fp32 synthetic gate; generation plus per-step ragged logits.
use crate::{IreeLlama, IreeRaggedLlama};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    prompt: Vec<i32>,
    generated: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

fn check_logits(got: &[f32], want: &[f32], context: &str) {
    assert_eq!(got.len(), want.len());
    let error = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{context}: non-finite logits"
    );
    assert!(error < 3e-4, "{context}: max_abs={error}");
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
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
