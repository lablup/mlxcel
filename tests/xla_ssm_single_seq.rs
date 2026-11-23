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

//! Real-IREE gates for the single-sequence recurrent-state handoff (epic #564,
//! issue #3339): prefill returns the Mamba2 state, the shim imports it into the
//! decode module's globals through `@state_import`, and every request starts
//! from a fresh state.
//!
//! The checkpoint is a synthetic two-layer Mamba2, so the gate checks the
//! state plumbing against itself: three different routes through prefill and
//! decode (full-prompt prefill, prompt prefix plus one decode, one-token
//! prefill plus decodes through the rest of the prompt) compute the same
//! recurrence and must agree token for token. The HF fp32 oracle for a real
//! checkpoint is `scripts/xla/validate_arch.sh --single-only`.
//!
//! Ignored in ordinary CI; run with the pinned IREE distribution:
//! `cargo test --release --features xla-iree,test-utils --test xla_ssm_single_seq -- --ignored`.

// `IreeLlama` and its test hooks are exported only under `test-support`, which
// the root `test-utils` feature enables.
#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use std::collections::HashMap;

use mlxcel_core::session::InferenceSession;
use mlxcel_xla::{IreeLlama, XlaInferenceSession, XlaReferenceEngine};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 24;
const HIDDEN: usize = 16;
const VOCAB: usize = 32;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const STATE: usize = 8;
const KERNEL: usize = 4;
const D_INNER: usize = HEADS * HEAD_DIM;
const CONV_DIM: usize = D_INNER + 2 * STATE;
const IN_PROJ: usize = 2 * D_INNER + 2 * STATE + HEADS;
const NEW_TOKENS: usize = 8;

struct TinyModel {
    dir: TempDir,
}

impl TinyModel {
    fn path(&self) -> &std::path::Path {
        self.dir.path()
    }
}

/// Deterministic values in `[-scale, scale]`, varied by `seed`.
fn deterministic_values(elements: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let raw = ((index * 7 + seed * 13) % 31) as f32 - 15.0;
            raw / 15.0 * scale
        })
        .collect()
}

/// Values in `[lo, hi]`, varied by `seed`.
fn ranged_values(elements: usize, seed: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| lo + (hi - lo) * (((index * 5 + seed * 3) % 11) as f32 / 10.0))
        .collect()
}

fn add_tensor(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: impl Into<String>,
    shape: &[usize],
    values: Vec<f32>,
) {
    assert_eq!(values.len(), shape.iter().product::<usize>());
    tensors.push((
        name.into(),
        shape.to_vec(),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    ));
}

/// A two-layer HF-layout `Mamba2ForCausalLM` with one group, a tied head and
/// conv biases, small enough to compile in seconds.
fn create_tiny_mamba2() -> TinyModel {
    let dir = tempfile::tempdir().expect("temporary model directory");
    std::fs::write(
        dir.path().join("config.json"),
        format!(
            r#"{{"model_type":"mamba2","hidden_size":{HIDDEN},"num_hidden_layers":{LAYERS},"vocab_size":{VOCAB},"num_heads":{HEADS},"head_dim":{HEAD_DIM},"state_size":{STATE},"n_groups":1,"conv_kernel":{KERNEL},"expand":2,"use_bias":false,"use_conv_bias":true,"tie_word_embeddings":true,"layer_norm_epsilon":1e-5,"rms_norm":true,"hidden_act":"silu","eos_token_id":0}}"#
        ),
    )
    .expect("model config");

    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        "backbone.embeddings.weight",
        &[VOCAB, HIDDEN],
        deterministic_values(VOCAB * HIDDEN, 1, 0.8),
    );
    add_tensor(
        &mut tensors,
        "backbone.norm_f.weight",
        &[HIDDEN],
        ranged_values(HIDDEN, 2, 0.6, 1.4),
    );
    for layer in 0..LAYERS {
        let p = format!("backbone.layers.{layer}.");
        let seed = 10 * (layer + 1);
        add_tensor(
            &mut tensors,
            format!("{p}norm.weight"),
            &[HIDDEN],
            ranged_values(HIDDEN, seed, 0.6, 1.4),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.in_proj.weight"),
            &[IN_PROJ, HIDDEN],
            deterministic_values(IN_PROJ * HIDDEN, seed + 1, 0.5),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.out_proj.weight"),
            &[HIDDEN, D_INNER],
            deterministic_values(HIDDEN * D_INNER, seed + 2, 0.4),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.conv1d.weight"),
            &[CONV_DIM, 1, KERNEL],
            deterministic_values(CONV_DIM * KERNEL, seed + 3, 0.6),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.conv1d.bias"),
            &[CONV_DIM],
            deterministic_values(CONV_DIM, seed + 4, 0.1),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.dt_bias"),
            &[HEADS],
            ranged_values(HEADS, seed + 5, -2.0, 0.0),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.A_log"),
            &[HEADS],
            ranged_values(HEADS, seed + 6, -1.0, 0.5),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.D"),
            &[HEADS],
            ranged_values(HEADS, seed + 7, 0.5, 1.5),
        );
        add_tensor(
            &mut tensors,
            format!("{p}mixer.norm.weight"),
            &[D_INNER],
            ranged_values(D_INNER, seed + 8, 0.6, 1.4),
        );
    }
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid tensor view"),
            )
        })
        .collect::<HashMap<_, _>>();
    safetensors::serialize_to_file(&views, None, &dir.path().join("model.safetensors"))
        .expect("model weights");
    TinyModel { dir }
}

fn device() -> String {
    std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string())
}

fn session(model: &TinyModel) -> XlaInferenceSession {
    XlaInferenceSession::load_with_context_capacity(model.path(), LAYERS, CAPACITY)
        .expect("load the Mamba2 session")
}

/// The CLI route: prefill the prompt prefix, decode the last prompt token.
fn cli_tokens(session: &mut XlaInferenceSession, prompt: &[i32]) -> Vec<i32> {
    session
        .generate_greedy(prompt, NEW_TOKENS, &[])
        .expect("session greedy generation")
}

/// Prefill only `prompt[0]` (`real_len = 1`), feed the rest of the prompt
/// through decode, then continue greedily: the state crosses the prefill to
/// decode boundary after one token and every later token is a decode step.
fn decode_through_prompt(engine: &mut IreeLlama, prompt: &[i32], new_tokens: usize) -> Vec<i32> {
    let mut next = engine
        .prefill_first(&prompt[..1])
        .expect("one-token prefill");
    for (position, &token) in prompt.iter().enumerate().skip(1) {
        next = engine
            .decode(token, position as i32)
            .expect("decode a prompt token");
    }
    let mut out = vec![next];
    let mut position = prompt.len() as i32;
    while out.len() < new_tokens {
        next = engine.decode(next, position).expect("greedy decode");
        position += 1;
        out.push(next);
    }
    out
}

const PROMPT_A: [i32; 6] = [3, 17, 5, 29, 11, 8];
const PROMPT_B: [i32; 5] = [21, 4, 4, 19, 2];

#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn prefill_to_decode_handoff_agrees_across_routes() {
    let model = create_tiny_mamba2();
    let mut reference =
        XlaReferenceEngine::load_with_context_capacity(model.path(), &device(), CAPACITY)
            .expect("load the reference engine");
    let mut engine =
        IreeLlama::load(model.path(), &device(), CAPACITY).expect("load the IREE engine");
    let mut session = session(&model);
    for prompt in [&PROMPT_A[..], &PROMPT_B[..]] {
        let full_prefill = reference
            .generate(prompt, NEW_TOKENS, &[])
            .expect("full-prompt prefill route");
        let cli = cli_tokens(&mut session, prompt);
        let decoded = decode_through_prompt(&mut engine, prompt, NEW_TOKENS);
        assert_eq!(cli, full_prefill, "prefix prefill + decode vs full prefill");
        assert_eq!(
            decoded, full_prefill,
            "real_len = 1 prefill + decodes vs full prefill"
        );
    }
}

#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn consecutive_requests_start_from_a_fresh_state() {
    let model = create_tiny_mamba2();
    let mut reused = session(&model);
    let first = cli_tokens(&mut reused, &PROMPT_A);
    let second = cli_tokens(&mut reused, &PROMPT_B);
    let fresh_first = cli_tokens(&mut session(&model), &PROMPT_A);
    let fresh_second = cli_tokens(&mut session(&model), &PROMPT_B);
    assert_eq!(first, fresh_first);
    assert_eq!(
        second, fresh_second,
        "the second request saw the first one's state"
    );

    // The check above is only meaningful if the model's output depends on the
    // history a leaked state would carry.
    let mut joined = PROMPT_A.to_vec();
    joined.extend_from_slice(&PROMPT_B);
    let continued = cli_tokens(&mut session(&model), &joined);
    assert_ne!(
        continued, fresh_second,
        "the synthetic model must be history-sensitive for the reset check to discriminate"
    );
}

#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn prefill_at_full_capacity_matches_the_decode_route() {
    let model = create_tiny_mamba2();
    let prompt: Vec<i32> = (0..CAPACITY)
        .map(|i| ((i * 11 + 3) % VOCAB) as i32)
        .collect();
    let mut engine =
        IreeLlama::load(model.path(), &device(), CAPACITY).expect("load the IREE engine");
    let full = engine
        .prefill_first(&prompt)
        .expect("real_len = capacity prefill");
    let decoded = decode_through_prompt(&mut engine, &prompt, 1);
    assert_eq!(vec![full], decoded);
}

#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn empty_prefix_and_refused_prefill_never_reuse_state() {
    let model = create_tiny_mamba2();
    let mut engine =
        IreeLlama::load(model.path(), &device(), CAPACITY).expect("load the IREE engine");
    assert_eq!(
        engine.test_state_status(),
        Some((false, false)),
        "no state before a request"
    );

    // A one-token prompt has an empty prefix: the session starts from the zero
    // state without a `real_len = 0` prefill, and must match a one-token
    // full-prompt prefill.
    let single = [PROMPT_A[0]];
    let mut reference =
        XlaReferenceEngine::load_with_context_capacity(model.path(), &device(), CAPACITY)
            .expect("load the reference engine");
    let want = reference
        .generate(&single, NEW_TOKENS, &[])
        .expect("one-token prefill route");
    let mut session = session(&model);
    cli_tokens(&mut session, &PROMPT_B);
    assert_eq!(cli_tokens(&mut session, &single), want);

    // The handoff: prefill leaves the state pending, the first decode imports it.
    engine.prefill_first(&PROMPT_A).expect("prefill");
    assert_eq!(engine.test_state_status(), Some((true, true)));
    engine.decode(1, PROMPT_A.len() as i32).expect("decode");
    assert_eq!(engine.test_state_status(), Some((true, false)));

    // `real_len = 0` is refused at the C boundary, and the refused request
    // drops the previous state so a decode cannot silently continue it.
    let tokens = vec![0; CAPACITY];
    let positions: Vec<i32> = (0..CAPACITY as i32).collect();
    assert!(
        engine
            .test_prefill_with_positions(&tokens, &positions, 0)
            .is_err()
    );
    assert_eq!(engine.test_state_status(), Some((false, false)));
    assert!(
        engine.decode(1, 0).is_err(),
        "decode after a refused prefill must fail"
    );
    assert!(
        engine.prefill_first(&[]).is_err(),
        "an empty full prompt is refused"
    );

    // A good request recovers.
    let mut recovered = vec![
        engine
            .prefill_first(&PROMPT_A)
            .expect("prefill after refusal"),
    ];
    let mut position = PROMPT_A.len() as i32;
    while recovered.len() < NEW_TOKENS {
        let next = engine
            .decode(*recovered.last().expect("token"), position)
            .expect("decode after refusal");
        position += 1;
        recovered.push(next);
    }
    let want_a = reference
        .generate(&PROMPT_A, NEW_TOKENS, &[])
        .expect("reference");
    assert_eq!(recovered, want_a);
}

/// A state reset that fails (the empty-prefix path of a one-token prompt) must
/// drop the previous request's state like a failed prefill does, so a decode
/// cannot silently continue that request.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn failed_state_reset_drops_the_previous_state() {
    let model = create_tiny_mamba2();
    let mut engine =
        IreeLlama::load(model.path(), &device(), CAPACITY).expect("load the IREE engine");
    engine.prefill_first(&PROMPT_A).expect("prefill");
    engine.decode(1, PROMPT_A.len() as i32).expect("decode");
    assert_eq!(engine.test_state_status(), Some((true, false)));

    engine.test_set_state_zero_failure(true);
    assert!(
        engine.prefill_seed(&[]).is_err(),
        "the forced zero-state failure must fail the reset"
    );
    assert_eq!(
        engine.test_state_status(),
        Some((false, false)),
        "a failed reset kept the previous request's state"
    );
    assert!(
        engine.decode(PROMPT_B[0], 0).is_err(),
        "decode after a failed reset must fail"
    );

    engine.test_set_state_zero_failure(false);
    engine
        .prefill_seed(&[])
        .expect("reset after clearing the hook");
    assert_eq!(engine.test_state_status(), Some((true, true)));
    let mut reference =
        XlaReferenceEngine::load_with_context_capacity(model.path(), &device(), CAPACITY)
            .expect("load the reference engine");
    let want = reference
        .generate(&PROMPT_B[..1], 1, &[])
        .expect("one-token reference");
    assert_eq!(
        vec![engine.decode(PROMPT_B[0], 0).expect("decode after reset")],
        want
    );
}

/// A request refused before it reaches the shim (an oversized prompt, checked
/// in Rust) must still drop the previous request's state, so neither the
/// engine's nor the session's decode can continue that earlier request.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn request_refused_in_rust_drops_the_previous_state() {
    let model = create_tiny_mamba2();
    let oversized = vec![PROMPT_B[0]; CAPACITY + 1];

    let mut engine =
        IreeLlama::load(model.path(), &device(), CAPACITY).expect("load the IREE engine");
    engine.prefill_first(&PROMPT_A).expect("prefill");
    engine.decode(1, PROMPT_A.len() as i32).expect("decode");
    assert_eq!(engine.test_state_status(), Some((true, false)));
    assert!(engine.prefill_seed(&oversized).is_err(), "oversized prefix");
    assert_eq!(
        engine.test_state_status(),
        Some((false, false)),
        "a prompt refused in Rust kept the previous request's state"
    );
    assert!(
        engine.decode(PROMPT_B[0], 0).is_err(),
        "decode after a refused request must fail"
    );
    engine.prefill_first(&PROMPT_A).expect("prefill");
    assert!(engine.prefill_first(&[]).is_err(), "empty prompt");
    assert_eq!(engine.test_state_status(), Some((false, false)));

    let mut session = session(&model);
    cli_tokens(&mut session, &PROMPT_A);
    assert!(InferenceSession::prefill(&mut session, &oversized).is_err());
    assert!(
        session.decode_step(PROMPT_B[0]).is_err(),
        "the session decoded on after a refused prefill"
    );
    cli_tokens(&mut session, &PROMPT_A);
    assert!(session.generate_greedy(&oversized, 1, &[]).is_err());
    assert!(
        session.decode_step(PROMPT_B[0]).is_err(),
        "the session decoded on after a refused generation"
    );
    assert_eq!(
        cli_tokens(&mut session, &PROMPT_A),
        cli_tokens(&mut self::session(&model), &PROMPT_A),
        "a good request recovers"
    );
}

/// The ragged, slot and embeddings entry points have no recurrent-state
/// contract (issue #3340): they admit only a KV-only bundle and refuse a hybrid
/// one, which would otherwise adopt a new KV next to the previous request's
/// state. Needs no device.
#[test]
fn kv_only_entry_points_refuse_any_recurrent_state() {
    assert_eq!(IreeLlama::test_kv_only_entries_refuse(true, 0), Some(false));
    assert_eq!(IreeLlama::test_kv_only_entries_refuse(false, 2), Some(true));
    assert_eq!(
        IreeLlama::test_kv_only_entries_refuse(true, 1),
        Some(true),
        "a hybrid bundle has no state contract on these entry points"
    );
    assert_eq!(IreeLlama::test_kv_only_entries_refuse(true, 2), Some(true));
}
