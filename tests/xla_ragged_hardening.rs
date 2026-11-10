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

//! Real-IREE gates for the ragged continuous-batching hardening in issue #2923.
//!
//! These belong to the root crate because its `xla-iree` build script owns the
//! whole-archive IREE runtime link recipe; they are ignored in ordinary CI and
//! need the pinned ROCm distribution and compiler plus the `test-utils` feature.

// The fixture types this gate drives (`IreeRaggedLlama` and the batch-KV
// counters) exist only under `test-support`, which the root `test-utils`
// feature enables. Without the gate, the plain `xla-iree` feature compile that
// CI runs with `--all-targets` fails to resolve those imports.
#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use std::collections::HashMap;

use mlxcel_xla::{
    IreeRaggedLlama, SampleParams, XlaBatchEngine, XlaReferenceEngine,
    batch_kv_allocation_counters, reset_batch_kv_allocation_counters,
};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 8;
const HIDDEN: usize = 8;
const VOCAB: usize = 32;
const MROPE_HIDDEN: usize = 12;

struct TinyModel {
    dir: TempDir,
}

impl TinyModel {
    fn path(&self) -> &std::path::Path {
        self.dir.path()
    }
}

fn deterministic_values(elements: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| (((index + seed * 17) % 29) as f32 - 14.0) * 0.003)
        .collect()
}

fn add_tensor(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: impl Into<String>,
    shape: &[usize],
    values: Vec<f32>,
) {
    tensors.push((
        name.into(),
        shape.to_vec(),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    ));
}

fn create_tiny_model() -> TinyModel {
    let dir = tempfile::tempdir().expect("temporary model directory");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"llama","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":32,"rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true}"#,
    )
    .expect("model config");

    let embeddings = deterministic_values(VOCAB * HIDDEN, 1);
    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        embeddings,
    );
    add_tensor(
        &mut tensors,
        "model.norm.weight",
        &[HIDDEN],
        vec![1.0; HIDDEN],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        add_tensor(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        );
        for (offset, suffix, shape) in [
            (2, "self_attn.q_proj.weight", vec![8, 8]),
            (3, "self_attn.k_proj.weight", vec![4, 8]),
            (4, "self_attn.v_proj.weight", vec![4, 8]),
            (5, "self_attn.o_proj.weight", vec![8, 8]),
            (6, "mlp.gate_proj.weight", vec![16, 8]),
            (7, "mlp.up_proj.weight", vec![16, 8]),
            (8, "mlp.down_proj.weight", vec![8, 16]),
        ] {
            add_tensor(
                &mut tensors,
                format!("{prefix}.{suffix}"),
                &shape,
                deterministic_values(shape.iter().product(), layer * 10 + offset),
            );
        }
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

/// A tiny Qwen2 checkpoint with M-RoPE sections, copied from the shape used by
/// `tests/xla_prepared_prefill.rs`. The ragged engine only takes the M-RoPE
/// decode path when the config declares `rope_scaling.mrope_section`.
fn create_tiny_mrope_model() -> TinyModel {
    let dir = tempfile::tempdir().expect("temporary M-RoPE model directory");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen2","hidden_size":12,"intermediate_size":24,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":6,"vocab_size":32,"rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true,"attention_bias":true,"hidden_act":"silu","rope_scaling":{"rope_type":"mrope","mrope_section":[1,1,1]}}"#,
    )
    .expect("M-RoPE model config");

    let embeddings = deterministic_values(VOCAB * MROPE_HIDDEN, 31);
    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[VOCAB, MROPE_HIDDEN],
        embeddings,
    );
    add_tensor(
        &mut tensors,
        "model.norm.weight",
        &[MROPE_HIDDEN],
        vec![1.0; MROPE_HIDDEN],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        add_tensor(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            &[MROPE_HIDDEN],
            vec![1.0; MROPE_HIDDEN],
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            &[MROPE_HIDDEN],
            vec![1.0; MROPE_HIDDEN],
        );
        for (offset, suffix, shape) in [
            (2, "self_attn.q_proj.weight", vec![12, 12]),
            (3, "self_attn.k_proj.weight", vec![6, 12]),
            (4, "self_attn.v_proj.weight", vec![6, 12]),
            (5, "self_attn.o_proj.weight", vec![12, 12]),
            (6, "mlp.gate_proj.weight", vec![24, 12]),
            (7, "mlp.up_proj.weight", vec![24, 12]),
            (8, "mlp.down_proj.weight", vec![12, 24]),
        ] {
            add_tensor(
                &mut tensors,
                format!("{prefix}.{suffix}"),
                &shape,
                deterministic_values(shape.iter().product(), layer * 20 + offset),
            );
        }
        for (offset, suffix, width) in [
            (11, "self_attn.q_proj.bias", 12),
            (12, "self_attn.k_proj.bias", 6),
            (13, "self_attn.v_proj.bias", 6),
        ] {
            add_tensor(
                &mut tensors,
                format!("{prefix}.{suffix}"),
                &[width],
                deterministic_values(width, layer * 20 + offset),
            );
        }
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
        .expect("M-RoPE model weights");
    TinyModel { dir }
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

/// Item 4: a store that dies after its K transfer leaves the slot invalid, an
/// active row on an invalid slot is rejected before IREE runs, and a successful
/// re-store makes the slot valid and reference-exact.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn failed_slot_store_leaves_the_slot_invalid_and_decode_rejects_it() {
    let fixture = create_tiny_model();
    let model = fixture.path();
    let device = device();
    let prompt = [1, 2, 3];
    let mut engine =
        IreeRaggedLlama::load(model, &device, 4, CAPACITY).expect("ragged engine at B=4");

    engine.test_set_slot_store_failure(true);
    engine
        .prefill_slot_logits(0, &prompt)
        .expect_err("the simulated V transfer failure must fail the slot prefill");
    engine.test_set_slot_store_failure(false);
    assert_eq!(
        engine.test_slot_valid(0),
        Some(false),
        "a store that failed after its K transfer must leave the slot invalid"
    );

    let modes = [0i32; 4];
    let mut tokens = [0i32; 4];
    let mut positions = [0i32; 4];
    let mut cache_len = [0i32; 4];
    tokens[0] = 7;
    positions[0] = prompt.len() as i32;
    cache_len[0] = prompt.len() as i32;
    let error = engine
        .decode_ragged_logits_with_modes(&modes, &tokens, &positions, &cache_len)
        .expect_err("an active row with an invalid KV slot must be rejected");
    assert!(
        error.contains("xla_llama_decode_ragged_logits failed"),
        "the C validator must fail the call, got: {error}"
    );

    let logits = engine
        .prefill_slot_logits(0, &prompt)
        .expect("a later successful store must revalidate the slot");
    assert_eq!(
        engine.test_slot_valid(0),
        Some(true),
        "both transfers succeeding must mark the slot valid"
    );
    // The identical row that was just rejected must now reach the graph, which
    // proves the rejection was the slot validity check and not a shape error.
    engine
        .decode_ragged_logits_with_modes(&modes, &tokens, &positions, &cache_len)
        .expect("the same row decodes once its KV store completed");
    let first = argmax(&logits);

    let mut reference = XlaReferenceEngine::load_with_context_capacity(model, &device, CAPACITY)
        .expect("reference engine");
    let expected = reference
        .generate(&prompt, 2, &[])
        .expect("two-token reference stream");
    assert_eq!(expected.len(), 2);
    assert_eq!(
        first, expected[0],
        "prefill logits must match the reference"
    );

    tokens[0] = first;
    let logits = engine
        .decode_ragged_logits_with_modes(&modes, &tokens, &positions, &cache_len)
        .expect("decode an active, valid row");
    assert_eq!(
        argmax(&logits[..VOCAB]),
        expected[1],
        "a valid slot's ragged decode must match the single-sequence reference"
    );
}

/// Item 2: an M-RoPE coordinate at the compiled capacity is rejected by the
/// Rust guard before the ragged graph can gather past the RoPE table, and a
/// coordinate one below it reaches the graph instead of being over-rejected.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn ragged_mrope_decode_rejects_a_coordinate_at_the_context_capacity() {
    let fixture = create_tiny_mrope_model();
    let mut engine = IreeRaggedLlama::load(fixture.path(), &device(), 4, CAPACITY)
        .expect("M-RoPE ragged engine at B=4");
    let modes = [0i32; 4];
    let mut tokens = [0i32; 4];
    let mut cache_len = [0i32; 4];
    let mut positions = [[0i32; 3]; 4];
    tokens[0] = 1;
    cache_len[0] = (CAPACITY - 1) as i32;
    positions[0] = [CAPACITY as i32; 3];
    let error = engine
        .decode_ragged_mrope_logits_with_modes(&modes, &tokens, &positions, &cache_len)
        .expect_err("a coordinate at context_capacity is an out-of-bounds gather");
    assert!(
        error.contains("outside context_capacity"),
        "the rejection must name the bound, got: {error}"
    );
    assert!(
        error.contains(&CAPACITY.to_string()),
        "the rejection must name the capacity, got: {error}"
    );

    positions[0] = [(CAPACITY - 1) as i32; 3];
    let error = engine
        .decode_ragged_mrope_logits_with_modes(&modes, &tokens, &positions, &cache_len)
        .expect_err("an in-range coordinate with no stored KV must fail inside the graph call");
    assert!(
        !error.contains("outside context_capacity"),
        "one row below the bound must not be rejected by the coordinate guard, got: {error}"
    );
}

/// Item 3: a reset at a batch size other than the compiled one is rejected
/// instead of surfacing later as an opaque graph shape mismatch inside IREE.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn ragged_reset_rejects_a_batch_size_the_graph_was_not_compiled_for() {
    let fixture = create_tiny_model();
    let mut engine = IreeRaggedLlama::load(fixture.path(), &device(), 4, CAPACITY)
        .expect("ragged engine at B=4");
    assert_ne!(
        engine.test_bind_and_reset(8, 4),
        0,
        "reset at B=4 must be rejected when the graph was bound at B=8"
    );
    assert_ne!(
        engine.test_bind_and_reset(0, 4),
        0,
        "reset before any bind must be rejected"
    );
    assert_eq!(
        engine.test_bind_and_reset(4, 4),
        0,
        "a reset matching the bound compiled B must still succeed"
    );
}

/// Item 1: the lazy rank-5 batch KV is zeroed on the device, with no host
/// mirror and no host-to-device copy of the cache on first admit.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn first_batch_admit_does_not_upload_a_host_kv_mirror() {
    let fixture = create_tiny_model();
    reset_batch_kv_allocation_counters();
    let mut engine =
        XlaBatchEngine::load_with_context_capacity(fixture.path(), 4, &device(), CAPACITY)
            .expect("batch engine");
    engine
        .submit(&[1, 2, 3], 2, SampleParams::greedy())
        .expect("submit request");
    let mut steps = 0;
    while !engine.is_idle() {
        steps += 1;
        assert!(steps < 64, "engine exceeded the safety step cap");
        engine.pump().expect("pump");
    }
    let (host_bytes, h2d_bytes) = batch_kv_allocation_counters();
    assert_eq!(
        host_bytes, 0,
        "the batch KV path must not allocate a host mirror of the rank-5 cache"
    );
    assert_eq!(
        h2d_bytes, 0,
        "the batch KV path must not upload the rank-5 cache from host memory"
    );
}
