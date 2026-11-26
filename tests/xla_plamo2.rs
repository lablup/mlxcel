// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0
//! PLaMo 2 gates (issue #3347) against the independent pure-PyTorch reference
//! in `spike/openxla/plamo2_oracle.py`, plus the real-checkpoint oracle.
//!
//! - `plamo2_synthetic_parity`: the hybrid fixture through single-sequence
//!   prefill + decode (tokens) and ragged prefill + decode (logits), then slot
//!   recycling.
//! - `plamo2_window_probe`: the 2048-token window through every graph flavor,
//!   plus the `W - 1` / `W + 1` controls, which must follow the exclusive and
//!   one-wider reference masks instead of the inclusive one.
#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use std::path::{Path, PathBuf};

use mlxcel_xla::{IreeLlama, IreeRaggedLlama};
use serde::Deserialize;

/// Logits tolerance for the hybrid fixture.
const TOL: f32 = 5e-3;
/// Window-probe tolerance. The closest variant pair (`age <= W` against
/// `age <= W + 1`) differs by about 5.9e-3 at the first step while the f32
/// graphs sit within 1e-5 of the float64 reference, so this separates them.
const WINDOW_TOL: f32 = 1e-3;

#[derive(Deserialize)]
struct Case {
    prompt: Vec<i32>,
    generated: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Trajectory {
    generated: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct WindowCase {
    prompt: Vec<i32>,
    inclusive: Trajectory,
    exclusive: Trajectory,
    plus: Trajectory,
    none: Trajectory,
}

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var("MLXCEL_PLAMO2_FIXTURES").expect("MLXCEL_PLAMO2_FIXTURES"))
}

fn device() -> String {
    std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".into())
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    assert!(got.iter().all(|x| x.is_finite()), "non-finite logits");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check_logits(got: &[f32], want: &[f32], context: &str) {
    let error = max_abs(got, want);
    assert!(error < TOL, "{context}: max_abs={error}");
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(got), argmax(want), "{context}");
}

/// Greedy single-sequence tokens through the sample prefill and decode graphs.
fn single_tokens(engine: &mut IreeLlama, prompt: &[i32], steps: usize) -> Vec<i32> {
    let mut token = engine.prefill_first(prompt).unwrap();
    let mut out = vec![token];
    for step in 1..steps {
        token = engine
            .decode(token, (prompt.len() + step - 1) as i32)
            .unwrap();
        out.push(token);
    }
    out
}

/// Ragged prefill + decode logits for up to `B` slots, following each slot's
/// own greedy tokens. Returns `[slot][step][vocab]`.
fn ragged_logits(
    engine: &mut IreeRaggedLlama,
    prompts: &[&[i32]],
    steps: usize,
    vocab: usize,
) -> Vec<Vec<Vec<f32>>> {
    let mut out: Vec<Vec<Vec<f32>>> = prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| vec![engine.prefill_slot_logits(slot, prompt).unwrap()])
        .collect();
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as i32
    };
    let b = engine.b_max();
    for step in 1..steps {
        let mut tokens = vec![0; b];
        let mut positions = vec![0; b];
        for (slot, prompt) in prompts.iter().enumerate() {
            tokens[slot] = argmax(out[slot].last().unwrap());
            positions[slot] = (prompt.len() + step - 1) as i32;
        }
        let modes = vec![0; b];
        let got = engine
            .decode_ragged_logits_with_modes(&modes, &tokens, &positions, &positions)
            .unwrap();
        for slot in 0..prompts.len() {
            out[slot].push(got[slot * vocab..(slot + 1) * vocab].to_vec());
        }
    }
    out
}

#[test]
#[ignore = "requires pinned IREE and fixtures from spike/openxla/plamo2_oracle.py"]
fn plamo2_synthetic_parity() {
    let path = fixtures().join("hybrid");
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(path.join("oracle.json")).unwrap()).unwrap();
    let device = device();
    let mut single = IreeLlama::load(&path, &device, 32).unwrap();
    for (i, case) in cases.iter().enumerate() {
        let got = single_tokens(&mut single, &case.prompt, case.generated.len());
        assert_eq!(got, case.generated, "case {i} single-sequence tokens");
    }
    let mut ragged = IreeRaggedLlama::load(&path, &device, 4, 32).unwrap();
    let vocab = ragged.vocab();
    let prompts: Vec<&[i32]> = cases.iter().map(|c| c.prompt.as_slice()).collect();
    let got = ragged_logits(&mut ragged, &prompts, 8, vocab);
    for (slot, case) in cases.iter().enumerate() {
        for (step, want) in case.logits.iter().enumerate() {
            check_logits(&got[slot][step], want, &format!("slot {slot} step {step}"));
        }
    }
    // Re-admission replaces both KV and recurrent state in an already used slot.
    for (slot, case) in cases.iter().rev().enumerate() {
        let got = ragged.prefill_slot_logits(slot, &case.prompt).unwrap();
        check_logits(&got, &case.logits[0], &format!("recycle slot {slot}"));
    }
}

/// Largest deviation of `got` from a reference trajectory.
fn deviation(got: &[Vec<f32>], want: &Trajectory) -> f32 {
    got.iter()
        .zip(&want.logits)
        .map(|(g, w)| max_abs(g, w))
        .fold(0.0, f32::max)
}

fn pick<'a>(c: &'a WindowCase, name: &str) -> &'a Trajectory {
    match name {
        "inclusive" => &c.inclusive,
        "exclusive" => &c.exclusive,
        "plus" => &c.plus,
        _ => &c.none,
    }
}

fn run_window(
    dir: &Path,
    cases: &[WindowCase],
    capacity: usize,
) -> (Vec<Vec<i32>>, Vec<Vec<Vec<f32>>>) {
    let device = device();
    let steps = cases[0].inclusive.generated.len();
    let mut single = IreeLlama::load(dir, &device, capacity).unwrap();
    let tokens = cases
        .iter()
        .map(|c| single_tokens(&mut single, &c.prompt, steps))
        .collect();
    drop(single);
    let mut ragged = IreeRaggedLlama::load(dir, &device, 4, capacity).unwrap();
    let vocab = ragged.vocab();
    let mut logits = Vec::new();
    for chunk in cases.chunks(4) {
        let prompts: Vec<&[i32]> = chunk.iter().map(|c| c.prompt.as_slice()).collect();
        logits.extend(ragged_logits(&mut ragged, &prompts, steps, vocab));
    }
    (tokens, logits)
}

#[test]
#[ignore = "requires pinned IREE and fixtures from spike/openxla/plamo2_oracle.py"]
fn plamo2_window_probe() {
    let root = fixtures();
    let cases: Vec<WindowCase> =
        serde_json::from_slice(&std::fs::read(root.join("window/oracle.json")).unwrap()).unwrap();
    let longest = cases.iter().map(|c| c.prompt.len()).max().unwrap();
    let capacity = (longest + cases[0].inclusive.generated.len()).next_multiple_of(32);
    for (dir, expect) in [
        ("window", "inclusive"),
        ("window_minus", "exclusive"),
        ("window_plus", "plus"),
    ] {
        let (tokens, logits) = run_window(&root.join(dir), &cases, capacity);
        let mut worst_expected: f32 = 0.0;
        for (i, case) in cases.iter().enumerate() {
            // The first-step logits do not depend on the followed trajectory,
            // so every variant is comparable there.
            let first: Vec<String> = ["inclusive", "exclusive", "plus", "none"]
                .iter()
                .map(|v| {
                    format!(
                        "{v}={:.3e}",
                        max_abs(&logits[i][0], &pick(case, v).logits[0])
                    )
                })
                .collect();
            eprintln!(
                "{dir} len {}: tokens {:?}; first-step |xla - variant|: {}",
                case.prompt.len(),
                tokens[i],
                first.join(" ")
            );
        }
        for (i, case) in cases.iter().enumerate() {
            let want = pick(case, expect);
            assert_eq!(
                tokens[i],
                want.generated,
                "{dir}: single-sequence tokens of the {}-token prompt follow {expect}",
                case.prompt.len()
            );
            assert_eq!(
                logits[i]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0 as i32
                    })
                    .collect::<Vec<_>>(),
                want.generated,
                "{dir}: ragged tokens of the {}-token prompt follow {expect}",
                case.prompt.len()
            );
            let error = deviation(&logits[i], want);
            worst_expected = worst_expected.max(error);
            assert!(
                error < WINDOW_TOL,
                "{dir} len {}: |xla-{expect}|={error}",
                case.prompt.len()
            );
            for other in ["inclusive", "exclusive", "plus", "none"] {
                if other == expect {
                    continue;
                }
                let other_t = pick(case, other);
                assert_ne!(
                    tokens[i],
                    other_t.generated,
                    "{dir} len {}: the probe cannot tell {expect} from {other}",
                    case.prompt.len()
                );
            }
        }
        eprintln!("{dir}: matches the {expect} mask; worst |logit delta| {worst_expected:.3e}");
    }
}

#[test]
#[ignore = "requires the pfnet/plamo-2-1b checkpoint and its 40-token MLX oracle"]
fn plamo2_original_checkpoint_oracle_tokens() {
    #[derive(Deserialize)]
    struct Oracle {
        prompt_ids: Vec<i32>,
        ref_token_ids: Vec<i32>,
    }
    let model = std::env::var("MLXCEL_PLAMO2_MODEL").expect("MLXCEL_PLAMO2_MODEL");
    let oracle = std::env::var("MLXCEL_PLAMO2_ORACLE").expect("MLXCEL_PLAMO2_ORACLE");
    let oracle: Oracle = serde_json::from_slice(&std::fs::read(oracle).unwrap()).unwrap();
    assert!(oracle.ref_token_ids.len() >= 40);
    let device = device();
    let mut engine = IreeLlama::load(Path::new(&model), &device, 64).unwrap();
    let got = single_tokens(&mut engine, &oracle.prompt_ids, oracle.ref_token_ids.len());
    let matched = got
        .iter()
        .zip(&oracle.ref_token_ids)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "PLaMo 2 {matched}/{} oracle tokens match on {device}",
        oracle.ref_token_ids.len()
    );
    assert_eq!(got, oracle.ref_token_ids);
}
