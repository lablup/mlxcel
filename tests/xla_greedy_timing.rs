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

//! Real-IREE gates for the greedy run's phase accounting in #3041.
//!
//! `mlxcel generate --profile` on the OpenXLA backend used to fold prefill into
//! the decode rate, so every reported decode figure depended on how many tokens
//! were requested. These belong to the root crate because its `xla-iree` build
//! script owns the whole-archive IREE runtime link recipe; they are ignored in
//! ordinary CI and need the pinned ROCm distribution and compiler plus the
//! `test-utils` feature.

#![cfg(all(feature = "xla-iree", feature = "test-utils"))]

use std::collections::HashMap;

use mlxcel_xla::XlaInferenceSession;
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

fn load_session(fixture: &TinyModel) -> XlaInferenceSession {
    XlaInferenceSession::load_with_context_capacity(fixture.path(), 2, CAPACITY)
        .expect("single-sequence session at the compiled capacity")
}

/// Issue #3041: the timed greedy run reports a non-zero prefill window and a
/// separate decode window, and it produces the same tokens as the untimed entry
/// point.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn greedy_run_reports_separate_prefill_and_decode_windows() {
    let fixture = create_tiny_model();
    let prompt = [1, 2, 3, 4];
    let mut timed = load_session(&fixture);
    let (tokens, timing) = timed
        .generate_streaming_greedy_timed(&prompt, 4, &[], |_| true)
        .expect("timed greedy run");
    assert_eq!(tokens.len(), 4, "the budget of four must be produced");
    assert!(
        timing.prefill_ms > 0.0,
        "the prefill window must be measured, not left at zero: {timing:?}"
    );
    assert!(
        timing.decode_ms > 0.0,
        "the decode window must be measured: {timing:?}"
    );

    let mut plain = load_session(&fixture);
    let plain_tokens = plain
        .generate_greedy(&prompt, 4, &[])
        .expect("untimed greedy run");
    assert_eq!(
        plain_tokens, tokens,
        "the timed entry point must not change the token stream"
    );
}

/// Issue #3041: a zero budget spends the prefill and reports no decode window,
/// which keeps the divide-by-zero guard honest at the CLI layer.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn zero_budget_reports_prefill_only() {
    let fixture = create_tiny_model();
    let mut session = load_session(&fixture);
    let (tokens, timing) = session
        .generate_streaming_greedy_timed(&[1, 2, 3], 0, &[], |_| true)
        .expect("zero-budget run");
    assert!(tokens.is_empty());
    assert!(timing.prefill_ms > 0.0, "prefill still runs: {timing:?}");
    assert_eq!(timing.decode_ms, 0.0, "no decode step runs: {timing:?}");
}
