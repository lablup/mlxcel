// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Real-IREE parity gate for token and prepared-embedding admission.
//!
//! This integration test belongs to the root crate because its `xla-iree`
//! build script owns the whole-archive IREE runtime link recipe. Run it with the
//! pinned distribution and compiler; it is ignored in ordinary CI.

use std::collections::HashMap;

use mlxcel::{
    OwnedTensor, PreparedAttentionBias, PreparedPositions, PreparedPrefill, PreparedTensorDType,
};
use mlxcel_xla::{EngineEvent, SampleParams, XlaBatchEngine, XlaInferenceSession};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 8;
const HIDDEN: usize = 8;
const VOCAB: usize = 32;

struct TinyModel {
    dir: TempDir,
    embeddings: Vec<f32>,
}

impl TinyModel {
    fn path(&self) -> &std::path::Path {
        self.dir.path()
    }

    fn prepared(&self, tokens: &[i32]) -> PreparedPrefill {
        let values = tokens
            .iter()
            .flat_map(|&token| {
                let row = token as usize * HIDDEN;
                self.embeddings[row..row + HIDDEN].iter().copied()
            })
            .collect::<Vec<_>>();
        PreparedPrefill::new(
            tokens.to_vec(),
            tensor_f32(&[1, tokens.len(), HIDDEN], &values),
            PreparedPositions::Sequential {
                start: 0,
                length: tokens.len(),
            },
            PreparedAttentionBias {
                tensor: tensor_f32(&[1, 1, 1, tokens.len()], &vec![0.0; tokens.len()]),
                causal: true,
            },
            Vec::new(),
        )
        .expect("valid prepared input")
    }
}

fn tensor_f32(shape: &[usize], values: &[f32]) -> OwnedTensor {
    OwnedTensor::new(
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        PreparedTensorDType::Float32,
        shape.to_vec(),
    )
    .expect("valid f32 tensor")
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
        embeddings.clone(),
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
    TinyModel { dir, embeddings }
}

#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn local_task_token_and_prepared_paths_are_exact_with_mixed_slot_reuse() {
    let fixture = create_tiny_model();
    let tokens = [1, 2, 3];
    let prepared = fixture.prepared(&tokens);

    let mut token_session =
        XlaInferenceSession::load_with_context_capacity(fixture.path(), 2, CAPACITY)
            .expect("token session");
    let mut prepared_session =
        XlaInferenceSession::load_with_context_capacity(fixture.path(), 2, CAPACITY)
            .expect("prepared session");
    let text_output = token_session
        .generate_greedy(&tokens, 2, &[])
        .expect("token generation");
    let prepared_output = prepared_session
        .generate_prepared_greedy(&prepared, 2, &[])
        .expect("prepared generation");
    assert_eq!(prepared_output, text_output);

    let mut batch =
        XlaBatchEngine::load_with_context_capacity(fixture.path(), 4, "local-task", CAPACITY)
            .expect("mixed batch");
    let cancelled = batch
        .submit_prepared(prepared.clone(), 4, SampleParams::greedy())
        .expect("queue cancellation candidate");
    assert!(batch.cancel(cancelled));
    assert!(batch.pump().expect("drain cancelled request").is_empty());

    let text = batch
        .submit(&tokens, 4, SampleParams::greedy())
        .expect("queue token request");
    let embedded = batch
        .submit_prepared(prepared, 4, SampleParams::greedy())
        .expect("queue prepared request");
    let mut text_tokens = Vec::new();
    let mut prepared_tokens = Vec::new();
    while !batch.is_idle() {
        for event in batch.pump().expect("mixed batch pump") {
            if let EngineEvent::Token { req_id, token } = event {
                if req_id == text {
                    text_tokens.push(token);
                } else if req_id == embedded {
                    prepared_tokens.push(token);
                }
            }
        }
    }
    assert_eq!(prepared_tokens, text_tokens);
    assert_eq!(text_tokens.len(), 4);

    let reused = batch
        .submit_prepared(fixture.prepared(&tokens), 2, SampleParams::greedy())
        .expect("reuse released slot");
    let events = batch.pump().expect("pump reused slot");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EngineEvent::Token { req_id, .. } if *req_id == reused))
    );
}
