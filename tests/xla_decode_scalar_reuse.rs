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

//! Real-IREE gates for the single-sequence decode-step scalar reuse in #3028.
//!
//! These belong to the root crate because its `xla-iree` build script owns the
//! whole-archive IREE runtime link recipe; they are ignored in ordinary CI and
//! need the pinned ROCm distribution and compiler plus the `test-utils` feature.

#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use std::collections::HashMap;

use mlxcel_xla::{IreeLlama, decode_scalar_counters, reset_decode_scalar_counters};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 8;
const HIDDEN: usize = 8;
const VOCAB: usize = 32;

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

fn load_engine(fixture: &TinyModel) -> IreeLlama {
    IreeLlama::load(fixture.path(), &device(), CAPACITY)
        .expect("single-sequence engine at the compiled capacity")
}

/// Decode a fixed token stream and return the tokens the graph produced.
fn decode_stream(engine: &mut IreeLlama, first: i32, steps: usize) -> Vec<i32> {
    let mut token = first;
    let mut produced = Vec::with_capacity(steps);
    for step in 0..steps {
        token = engine
            .decode(token, 3 + step as i32)
            .unwrap_or_else(|error| panic!("decode at step {step}: {error}"));
        produced.push(token);
    }
    produced
}

/// Issue #3028: the single-sequence decode step reuses one set of persistent
/// scalar views for its token, positions, and cache length, plus one persistent
/// readback slot. Before this change each step allocated three pinned staging
/// input views and one readback staging buffer, and on gfx1151 a pinned free
/// alone costs ~120 us. A non-zero allocation count here means the per-step
/// path came back.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn decode_steps_reuse_the_persistent_scalars() {
    let fixture = create_tiny_model();
    let mut engine = load_engine(&fixture);
    let first = engine
        .prefill_first(&[1, 2, 3])
        .expect("prefill must return its first token");
    // The persistent scalars are created by the first decode through a
    // dedicated allocator, so even that step stays off the per-step path.
    reset_decode_scalar_counters();
    let produced = decode_stream(&mut engine, first, 3);
    assert!(produced.iter().all(|&token| token >= 0));
    assert_eq!(
        decode_scalar_counters(),
        (0, 3),
        "three decode steps must allocate no staging buffers and read the \
         token back through the persistent device-to-device copy"
    );
    assert_eq!(
        engine.test_decode_step_scalars_state(),
        1,
        "the persistent scalars must be active after the first decode"
    );
}

/// Issue #3028: a device whose allocator cannot serve the host-local,
/// device-visible heap degrades to today's per-step allocation path instead of
/// failing. The forced fallback must produce the same token stream as the
/// reuse path.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn forced_fallback_decodes_the_same_token_stream() {
    let fixture = create_tiny_model();

    let mut reuse = load_engine(&fixture);
    let reuse_first = reuse
        .prefill_first(&[1, 2, 3])
        .expect("prefill must return its first token");
    let reuse_tokens = decode_stream(&mut reuse, reuse_first, 3);
    assert_eq!(reuse.test_decode_step_scalars_state(), 1);

    let mut fallback = load_engine(&fixture);
    let fallback_first = fallback
        .prefill_first(&[1, 2, 3])
        .expect("prefill must return its first token");
    assert_eq!(
        fallback_first, reuse_first,
        "both engines must agree on prefill"
    );
    fallback.test_force_step_scalars_unsupported(true);
    reset_decode_scalar_counters();
    let fallback_tokens = decode_stream(&mut fallback, fallback_first, 3);
    assert_eq!(
        fallback_tokens, reuse_tokens,
        "the fallback path must decode the same tokens"
    );
    assert_eq!(
        decode_scalar_counters(),
        (9, 0),
        "the forced fallback must allocate three input views per step and skip \
         the persistent readback"
    );
    assert_eq!(fallback.test_decode_step_scalars_state(), -1);
}

/// Issue #3028: a context that only prefills and then frees must not allocate
/// the persistent scalars, and freeing must stay clean.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn prefill_only_context_never_allocates_the_scalars() {
    let fixture = create_tiny_model();
    let mut engine = load_engine(&fixture);
    engine
        .prefill_seed(&[1, 2, 3])
        .expect("prefill without decode");
    assert_eq!(
        engine.test_decode_step_scalars_state(),
        0,
        "a context that never decodes must not create the persistent scalars"
    );
    drop(engine);
}
