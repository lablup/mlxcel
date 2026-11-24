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

//! Real-IREE gates for the per-slot recurrent state of the continuous-batching
//! engine (epic #564, issue #3344): each slot's hybrid KV and recurrent state is admitted by its
//! prefill, carried row by row through the ragged decode, and replaced whole
//! when the slot is recycled. Every batched stream must equal the request's
//! single-sequence reference (`IreeLlama`, the CLI path).
//!
//! The checkpoint is a synthetic five-layer Nemotron-H (M*-M-); `scripts/xla/validate_arch.sh`
//! runs the same gate on a real one against the HF fp32 oracle.
//!
//! Ignored in ordinary CI; run with the pinned IREE distribution:
//! `cargo test --release --features xla-iree,test-utils --test xla_hybrid_nemotron_h -- --ignored`.

#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use std::collections::HashMap;

use mlxcel_xla::{
    EngineEvent, IreeRaggedLlama, SampleParams, XlaBatchEngine, XlaReferenceEngine,
    batch_state_allocation_counters, reset_batch_state_allocation_counters,
};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 32;
const B: usize = 4;
const HIDDEN: usize = 16;
const VOCAB: usize = 32;
const SSM_LAYERS: usize = 2;
const HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const STATE: usize = 8;
const KERNEL: usize = 4;
const D_INNER: usize = HEADS * HEAD_DIM;
const CONV_DIM: usize = D_INNER + 2 * 2 * STATE;
const IN_PROJ: usize = D_INNER + CONV_DIM + HEADS;
/// Bytes of one sequence's state: `ssm_h [L, H, P, N]` plus `ssm_conv
/// [L, K-1, conv_dim]`, f32.
const STATE_ROW_BYTES: u64 =
    (4 * (SSM_LAYERS * HEADS * HEAD_DIM * STATE + SSM_LAYERS * (KERNEL - 1) * CONV_DIM)) as u64;

/// Distinct prompts of different lengths; the EOS id (31) never appears in
/// them.
const PROMPTS: [&[i32]; 7] = [
    &[3, 17, 5, 29, 11, 8],
    &[21, 4, 4, 19, 2],
    &[9],
    &[14, 1, 27, 6, 6, 12, 25, 7, 18],
    &[2, 30, 13],
    &[22, 5, 16, 16, 9, 1, 3],
    &[11, 28],
];

struct TinyModel {
    dir: TempDir,
}

impl TinyModel {
    fn path(&self) -> &std::path::Path {
        self.dir.path()
    }
}

fn deterministic_values(elements: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| (((index * 7 + seed * 13) % 31) as f32 - 15.0) / 15.0 * scale)
        .collect()
}

fn ranged_values(elements: usize, seed: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| lo + (hi - lo) * (((index * 5 + seed * 3) % 11) as f32 / 10.0))
        .collect()
}

/// HF-layout hybrid with two SSM norm groups, two KV heads, a wider attention
/// projection than the hidden width, and FFN-only relu2 blocks.
fn create_tiny_nemotron_h() -> TinyModel {
    let dir = tempfile::tempdir().expect("temporary model directory");
    std::fs::write(
        dir.path().join("config.json"),
        format!(
            r#"{{"model_type":"nemotron_h","hybrid_override_pattern":"M*-M-","num_attention_heads":4,"num_key_value_heads":2,"attention_head_dim":8,"intermediate_size":32,"hidden_size":{HIDDEN},"num_hidden_layers":5,"vocab_size":{VOCAB},"mamba_num_heads":{HEADS},"mamba_head_dim":{HEAD_DIM},"ssm_state_size":{STATE},"n_groups":2,"conv_kernel":{KERNEL},"chunk_size":128,"expand":2,"use_bias":false,"use_conv_bias":true,"tie_word_embeddings":true,"layer_norm_epsilon":1e-5,"rms_norm":true,"hidden_act":"silu","eos_token_id":31}}"#
        ),
    )
    .expect("model config");
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "backbone.embeddings.weight".into(),
            vec![VOCAB, HIDDEN],
            deterministic_values(VOCAB * HIDDEN, 1, 0.8),
        ),
        (
            "backbone.norm_f.weight".into(),
            vec![HIDDEN],
            ranged_values(HIDDEN, 2, 0.6, 1.4),
        ),
    ];
    for layer in 0..5 {
        if layer == 1 || layer == 2 || layer == 4 {
            let p = format!("backbone.layers.{layer}.");
            tensors.push((
                format!("{p}norm.weight"),
                vec![HIDDEN],
                ranged_values(HIDDEN, layer + 40, 0.6, 1.4),
            ));
            let projections = if layer == 1 {
                vec![
                    ("q_proj", 32, HIDDEN),
                    ("k_proj", 16, HIDDEN),
                    ("v_proj", 16, HIDDEN),
                    ("o_proj", HIDDEN, 32),
                ]
            } else {
                vec![("up_proj", 32, HIDDEN), ("down_proj", HIDDEN, 32)]
            };
            for (i, (name, output, input)) in projections.into_iter().enumerate() {
                tensors.push((
                    format!("{p}mixer.{name}.weight"),
                    vec![output, input],
                    deterministic_values(output * input, layer * 10 + i, 0.15),
                ));
            }
            continue;
        }
        let p = format!("backbone.layers.{layer}.");
        let s = 10 * (layer + 1);
        tensors.extend([
            (
                format!("{p}norm.weight"),
                vec![HIDDEN],
                ranged_values(HIDDEN, s, 0.6, 1.4),
            ),
            (
                format!("{p}mixer.in_proj.weight"),
                vec![IN_PROJ, HIDDEN],
                deterministic_values(IN_PROJ * HIDDEN, s + 1, 0.5),
            ),
            (
                format!("{p}mixer.out_proj.weight"),
                vec![HIDDEN, D_INNER],
                deterministic_values(HIDDEN * D_INNER, s + 2, 0.4),
            ),
            (
                format!("{p}mixer.conv1d.weight"),
                vec![CONV_DIM, 1, KERNEL],
                deterministic_values(CONV_DIM * KERNEL, s + 3, 0.6),
            ),
            (
                format!("{p}mixer.conv1d.bias"),
                vec![CONV_DIM],
                deterministic_values(CONV_DIM, s + 4, 0.1),
            ),
            (
                format!("{p}mixer.dt_bias"),
                vec![HEADS],
                ranged_values(HEADS, s + 5, -2.0, 0.0),
            ),
            (
                format!("{p}mixer.A_log"),
                vec![HEADS],
                ranged_values(HEADS, s + 6, -1.0, 0.5),
            ),
            (
                format!("{p}mixer.D"),
                vec![HEADS],
                ranged_values(HEADS, s + 7, 0.5, 1.5),
            ),
            (
                format!("{p}mixer.norm.weight"),
                vec![D_INNER],
                ranged_values(D_INNER, s + 8, 0.6, 1.4),
            ),
        ]);
    }
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(name, shape, values)| {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            let raw = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (name, shape, raw)
        })
        .collect();
    let views = bytes
        .iter()
        .map(|(name, shape, raw)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), raw).expect("valid tensor view"),
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

fn argmax(values: &[f32]) -> i32 {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best as i32
}

/// The single-sequence reference stream (the CLI's `IreeLlama` path) for each
/// `(prompt, budget)`.
fn references(model: &TinyModel, requests: &[(&[i32], usize)], eos: &[i32]) -> Vec<Vec<i32>> {
    let mut reference =
        XlaReferenceEngine::load_with_context_capacity(model.path(), &device(), CAPACITY)
            .expect("load the reference engine");
    requests
        .iter()
        .map(|(prompt, cap)| reference.generate(prompt, *cap, eos).expect("reference"))
        .collect()
}

/// Drive `requests` through the batch engine to completion; the per-request
/// token streams in submission order.
fn run_batch(engine: &mut XlaBatchEngine, requests: &[(&[i32], usize)]) -> Vec<Vec<i32>> {
    let ids: Vec<u64> = requests
        .iter()
        .map(|(prompt, cap)| {
            engine
                .submit(prompt, *cap, SampleParams::greedy())
                .expect("submit")
        })
        .collect();
    let mut streams: HashMap<u64, Vec<i32>> = HashMap::new();
    let mut steps = 0;
    while !engine.is_idle() {
        steps += 1;
        assert!(steps < 512, "engine exceeded the safety step cap");
        for event in engine.pump().expect("pump") {
            if let EngineEvent::Token { req_id, token } = event {
                streams.entry(req_id).or_default().push(token);
            }
        }
    }
    ids.iter()
        .map(|id| streams.remove(id).unwrap_or_default())
        .collect()
}

/// More requests than slots with staggered budgets: later requests are admitted
/// mid-stream into slots whose previous occupant just finished, next to rows
/// that are still decoding. Each stream must equal its own reference, so a
/// recycled slot never carries its previous occupant's state and no row
/// disturbs another.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn mid_stream_admit_into_a_recycled_slot_matches_the_reference() {
    let model = create_tiny_nemotron_h();
    let requests: Vec<(&[i32], usize)> = PROMPTS
        .iter()
        .enumerate()
        .map(|(i, prompt)| (*prompt, 3 + (i * 5) % 11))
        .collect();
    assert!(requests.len() > B, "the workload must recycle slots");
    let mut engine =
        XlaBatchEngine::load_with_context_capacity(model.path(), B, &device(), CAPACITY)
            .expect("load the batch engine");
    let eos = engine.eos_token_ids().to_vec();
    let want = references(&model, &requests, &eos);
    let got = run_batch(&mut engine, &requests);
    for (index, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(got, want, "request {index} diverged from its reference");
    }
}

/// The same requests give the same ids whether they share a half-full or a
/// full batch, so a row's output never depends on how many peers it has.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn half_full_and_full_batches_give_identical_per_request_ids() {
    let model = create_tiny_nemotron_h();
    let full: Vec<(&[i32], usize)> = PROMPTS[..B].iter().map(|prompt| (*prompt, 12)).collect();
    let mut engine =
        XlaBatchEngine::load_with_context_capacity(model.path(), B, &device(), CAPACITY)
            .expect("load the batch engine");
    let half_ids = run_batch(&mut engine, &full[..B / 2]);
    let full_ids = run_batch(&mut engine, &full);
    assert_eq!(half_ids[..], full_ids[..B / 2]);
    let eos = engine.eos_token_ids().to_vec();
    assert_eq!(full_ids, references(&model, &full, &eos));
}

/// One decode row: `token` at the slot's current length, every other row
/// inactive (all zeros).
fn one_row(slot: usize, token: i32, len: usize) -> ([i32; B], [i32; B], [i32; B]) {
    let mut tokens = [0i32; B];
    let mut positions = [0i32; B];
    tokens[slot] = token;
    positions[slot] = len as i32;
    (tokens, positions, positions)
}

/// Greedy-decode slot `slot` for `steps` tokens after its prefill produced
/// `first`, with every other row inactive.
fn decode_slot(
    engine: &mut IreeRaggedLlama,
    slot: usize,
    first: i32,
    prompt_len: usize,
    steps: usize,
) -> Vec<i32> {
    let mut out = vec![first];
    while out.len() < steps {
        let (tokens, positions, lens) = one_row(
            slot,
            *out.last().expect("token"),
            prompt_len + out.len() - 1,
        );
        let logits = engine
            .decode_ragged_logits_with_modes(&[0; B], &tokens, &positions, &lens)
            .expect("decode the slot");
        out.push(argmax(&logits[slot * VOCAB..(slot + 1) * VOCAB]));
    }
    out
}

/// A store that dies after the first state tensor (`ssm_h` written, `ssm_conv`
/// not) leaves the slot invalid even though it previously held a complete
/// state, an active row on it is refused before IREE runs, and the next admit
/// stores a whole fresh state and decodes reference-exact.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn torn_state_store_leaves_the_slot_invalid_and_the_next_admit_recovers() {
    let model = create_tiny_nemotron_h();
    let (prompt_a, prompt_b) = (PROMPTS[0], PROMPTS[1]);
    let mut engine =
        IreeRaggedLlama::load(model.path(), &device(), B, CAPACITY).expect("ragged engine at B=4");
    engine
        .prefill_slot_logits(0, prompt_b)
        .expect("seed slot 0");
    assert_eq!(engine.test_slot_valid(0), Some(true));

    engine.test_set_slot_store_failure(true);
    engine
        .prefill_slot_logits(0, prompt_a)
        .expect_err("the simulated state transfer failure must fail the prefill");
    engine.test_set_slot_store_failure(false);
    assert_eq!(
        engine.test_slot_valid(0),
        Some(false),
        "torn store is valid"
    );
    let (tokens, positions, lens) = one_row(0, 7, prompt_a.len());
    let error = engine
        .decode_ragged_logits_with_modes(&[0; B], &tokens, &positions, &lens)
        .expect_err("an active row on a torn slot must be refused");
    assert!(
        error.contains("xla_llama_decode_ragged_logits failed"),
        "{error}"
    );

    let logits = engine
        .prefill_slot_logits(0, prompt_a)
        .expect("the next admit recovers the slot");
    assert_eq!(engine.test_slot_valid(0), Some(true));
    let got = decode_slot(&mut engine, 0, argmax(&logits), prompt_a.len(), 8);
    assert_eq!(got, references(&model, &[(prompt_a, 8)], &[])[0]);
}

/// A request refused in Rust before the shim runs (an empty or oversized
/// prompt) invalidates the slot it was meant for, and a failed ragged decode
/// invalidates every slot, so no row can continue a state it did not finish.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn refused_requests_and_failed_decodes_invalidate_slot_state() {
    let model = create_tiny_nemotron_h();
    let mut engine =
        IreeRaggedLlama::load(model.path(), &device(), B, CAPACITY).expect("ragged engine at B=4");
    for slot in 0..3 {
        engine
            .prefill_slot_logits(slot, PROMPTS[slot])
            .expect("seed a slot");
    }
    engine
        .prefill_slot_logits(0, &[])
        .expect_err("an empty prompt is refused");
    engine
        .prefill_slot_logits(1, &[5; CAPACITY + 1])
        .expect_err("an oversized prompt is refused");
    assert_eq!(engine.test_slot_valid(0), Some(false), "empty prompt");
    assert_eq!(engine.test_slot_valid(1), Some(false), "oversized prompt");
    assert_eq!(
        engine.test_slot_valid(2),
        Some(true),
        "other slots untouched"
    );

    // A step that fails (an out-of-vocabulary row, refused by the shim) leaves
    // no slot valid on a recurrent bundle.
    let (mut tokens, positions, lens) = one_row(2, VOCAB as i32, PROMPTS[2].len());
    engine
        .decode_ragged_logits_with_modes(&[0; B], &tokens, &positions, &lens)
        .expect_err("an out-of-vocabulary token is refused");
    assert_eq!(engine.test_slot_valid(2), Some(false), "failed decode");
    tokens[2] = 1;
    engine
        .decode_ragged_logits_with_modes(&[0; B], &tokens, &positions, &lens)
        .expect_err("a row after a failed decode is refused until re-prefilled");

    let logits = engine.prefill_slot_logits(2, PROMPTS[3]).expect("re-admit");
    let got = decode_slot(&mut engine, 2, argmax(&logits), PROMPTS[3].len(), 6);
    assert_eq!(got, references(&model, &[(PROMPTS[3], 6)], &[])[0]);
}

/// The per-slot state set is allocated zeroed once per reset, on the first
/// admit, sized `B` rows of one sequence's state, and not again by later
/// admits or decodes until the next reset.
#[test]
#[ignore = "requires the pinned IREE distribution and compiler"]
fn state_allocation_counters_show_one_allocation_per_reset() {
    let model = create_tiny_nemotron_h();
    let mut engine =
        IreeRaggedLlama::load(model.path(), &device(), B, CAPACITY).expect("ragged engine at B=4");
    reset_batch_state_allocation_counters();
    assert_eq!(
        batch_state_allocation_counters(),
        (0, 0),
        "lazy until admit"
    );
    let mut firsts = Vec::new();
    for slot in 0..B {
        let logits = engine
            .prefill_slot_logits(slot, PROMPTS[slot])
            .expect("admit");
        firsts.push(argmax(&logits));
    }
    let mut tokens = [0i32; B];
    let mut lens = [0i32; B];
    for slot in 0..B {
        tokens[slot] = firsts[slot];
        lens[slot] = PROMPTS[slot].len() as i32;
    }
    engine
        .decode_ragged_logits_with_modes(&[0; B], &tokens, &lens, &lens)
        .expect("decode a full batch");
    let per_set = B as u64 * STATE_ROW_BYTES;
    assert_eq!(batch_state_allocation_counters(), (1, per_set));

    assert_eq!(engine.test_bind_and_reset(B as i32, B as i32), 0, "reset");
    assert_eq!(
        engine.test_slot_valid(0),
        Some(false),
        "a reset clears slots"
    );
    engine.prefill_slot_logits(0, PROMPTS[0]).expect("admit");
    engine.prefill_slot_logits(1, PROMPTS[1]).expect("admit");
    assert_eq!(batch_state_allocation_counters(), (2, 2 * per_set));
}
