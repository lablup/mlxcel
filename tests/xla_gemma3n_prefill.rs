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

//! Real-IREE Gemma3n token/dense-PLE equivalence and mixed-slot lifecycle gate.

#![cfg(feature = "xla-iree")]

use std::collections::HashMap;

use mlxcel::{
    OwnedTensor, PreparedAttentionBias, PreparedPositions, PreparedPrefill, PreparedTensorDType,
};
use mlxcel_xla::{
    EngineEvent, Gemma3nDensePle, Gemma3nPreparedPrefill, SampleParams, XlaBatchEngine,
    XlaInferenceSession,
};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 8;
const HIDDEN: usize = 8;
const LAYERS: usize = 4;
const PLE_HIDDEN: usize = 2;
const PLE_WIDTH: usize = LAYERS * PLE_HIDDEN;
const VOCAB: usize = 32;
const PER_LAYER_VOCAB: usize = 24;

struct TinyGemma3n {
    dir: TempDir,
    embeddings: Vec<f32>,
    token_ple: Vec<f32>,
    model_projection: Vec<f32>,
}

impl TinyGemma3n {
    fn path(&self) -> &std::path::Path {
        self.dir.path()
    }

    fn prepared(&self, tokens: &[i32]) -> Gemma3nPreparedPrefill {
        let scale = (HIDDEN as f32).sqrt();
        let embeddings = tokens
            .iter()
            .flat_map(|&token| {
                let row = token as usize * HIDDEN;
                self.embeddings[row..row + HIDDEN]
                    .iter()
                    .map(move |value| value * scale)
            })
            .collect::<Vec<_>>();
        let prepared = PreparedPrefill::new(
            tokens.to_vec(),
            tensor_f32(&[1, tokens.len(), HIDDEN], &embeddings),
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
        .expect("valid prepared embeddings");

        let mut dense = vec![0.0; CAPACITY * PLE_WIDTH];
        for (position, &token) in tokens.iter().enumerate() {
            let base = &embeddings[position * HIDDEN..(position + 1) * HIDDEN];
            let mut projected = [0.0; PLE_WIDTH];
            for (output, value) in projected.iter_mut().enumerate() {
                *value = (0..HIDDEN)
                    .map(|input| self.model_projection[output * HIDDEN + input] * base[input])
                    .sum::<f32>()
                    / (HIDDEN as f32).sqrt();
            }
            for layer in 0..LAYERS {
                let start = layer * PLE_HIDDEN;
                rms_norm(&mut projected[start..start + PLE_HIDDEN], 1e-6);
            }
            let mut token_row = [0.0; PLE_WIDTH];
            if token >= 0 && (token as usize) < PER_LAYER_VOCAB {
                let start = token as usize * PLE_WIDTH;
                for (output, value) in token_row.iter_mut().enumerate() {
                    *value = self.token_ple[start + output] * (PLE_HIDDEN as f32).sqrt();
                }
            }
            for output in 0..PLE_WIDTH {
                dense[position * PLE_WIDTH + output] =
                    (projected[output] + token_row[output]) * std::f32::consts::FRAC_1_SQRT_2;
            }
        }
        let dense_ple =
            Gemma3nDensePle::new(dense, CAPACITY, LAYERS, PLE_HIDDEN).expect("valid dense PLE");
        Gemma3nPreparedPrefill::new(prepared, dense_ple).expect("aligned Gemma3n request")
    }
}

fn rms_norm(row: &mut [f32], eps: f32) {
    let mean = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
    let scale = (mean + eps).sqrt().recip();
    for value in row {
        *value *= scale;
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
        .map(|index| (((index + seed * 19) % 31) as f32 - 15.0) * 0.002)
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
        values.into_iter().flat_map(f32::to_le_bytes).collect(),
    ));
}

fn create_tiny_gemma3n() -> TinyGemma3n {
    let dir = tempfile::tempdir().expect("temporary Gemma3n model");
    let config = serde_json::json!({
        "model_type": "gemma3n",
        "text_config": {
            "model_type": "gemma3n_text",
            "hidden_size": HIDDEN,
            "max_position_embeddings": 4096,
            "intermediate_size": [12, 12, 12, 12],
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "vocab_size_per_layer_input": PER_LAYER_VOCAB,
            "hidden_size_per_layer_input": PLE_HIDDEN,
            "layer_types": [
                "sliding_attention", "full_attention",
                "sliding_attention", "full_attention"
            ],
            "activation_sparsity_pattern": [0.5, 0.0, 0.0, 0.0],
            "sliding_window": 4,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "final_logit_softcapping": 30.0,
            "num_kv_shared_layers": 2,
            "altup_num_inputs": 2,
            "altup_active_idx": 0,
            "altup_coef_clip": 120.0,
            "altup_correct_scale": true,
            "laurel_rank": 2,
            "tie_word_embeddings": true
        }
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    let root = "model.language_model";
    let embeddings = deterministic_values(VOCAB * HIDDEN, 1);
    let token_ple = deterministic_values(PER_LAYER_VOCAB * PLE_WIDTH, 2);
    let model_projection = deterministic_values(PLE_WIDTH * HIDDEN, 3);
    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        format!("{root}.embed_tokens.weight"),
        &[VOCAB, HIDDEN],
        embeddings.clone(),
    );
    add_tensor(
        &mut tensors,
        format!("{root}.embed_tokens_per_layer.weight"),
        &[PER_LAYER_VOCAB, PLE_WIDTH],
        token_ple.clone(),
    );
    add_tensor(
        &mut tensors,
        format!("{root}.per_layer_model_projection.weight"),
        &[PLE_WIDTH, HIDDEN],
        model_projection.clone(),
    );
    add_tensor(
        &mut tensors,
        format!("{root}.per_layer_projection_norm.weight"),
        &[PLE_HIDDEN],
        vec![1.0; PLE_HIDDEN],
    );
    add_tensor(
        &mut tensors,
        format!("{root}.norm.weight"),
        &[HIDDEN],
        vec![1.0; HIDDEN],
    );
    add_tensor(
        &mut tensors,
        format!("{root}.altup_projections.0.weight"),
        &[HIDDEN, HIDDEN],
        deterministic_values(HIDDEN * HIDDEN, 4),
    );
    add_tensor(
        &mut tensors,
        format!("{root}.altup_unembed_projections.0.weight"),
        &[HIDDEN, HIDDEN],
        deterministic_values(HIDDEN * HIDDEN, 5),
    );
    for layer in 0..LAYERS {
        add_layer_tensors(&mut tensors, root, layer);
    }
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid tensor"),
            )
        })
        .collect::<HashMap<_, _>>();
    safetensors::serialize_to_file(&views, None, &dir.path().join("model.safetensors"))
        .expect("Gemma3n weights");
    TinyGemma3n {
        dir,
        embeddings,
        token_ple,
        model_projection,
    }
}

fn add_layer_tensors(tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>, root: &str, layer: usize) {
    let prefix = format!("{root}.layers.{layer}");
    let add = |tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
               suffix: &str,
               shape: &[usize],
               seed: usize| {
        add_tensor(
            tensors,
            format!("{prefix}.{suffix}"),
            shape,
            deterministic_values(shape.iter().product(), layer * 31 + seed),
        );
    };
    add_tensor(
        tensors,
        format!("{prefix}.altup.correct_output_scale"),
        &[HIDDEN],
        vec![1.0; HIDDEN],
    );
    add(tensors, "altup.correction_coefs.weight", &[2, 2], 2);
    add(tensors, "altup.modality_router.weight", &[2, HIDDEN], 3);
    add_tensor(
        tensors,
        format!("{prefix}.altup.router_norm.weight"),
        &[HIDDEN],
        vec![1.0; HIDDEN],
    );
    add(tensors, "altup.prediction_coefs.weight", &[4, 2], 5);
    add(tensors, "laurel.linear_left.weight", &[2, HIDDEN], 6);
    add(tensors, "laurel.linear_right.weight", &[HIDDEN, 2], 7);
    for suffix in [
        "laurel.post_laurel_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ] {
        add_tensor(
            tensors,
            format!("{prefix}.{suffix}"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        );
    }
    add(tensors, "self_attn.q_proj.weight", &[8, HIDDEN], 13);
    if layer < 2 {
        add(tensors, "self_attn.k_proj.weight", &[4, HIDDEN], 14);
        add(tensors, "self_attn.v_proj.weight", &[4, HIDDEN], 15);
    }
    add(tensors, "self_attn.o_proj.weight", &[HIDDEN, 8], 16);
    add_tensor(
        tensors,
        format!("{prefix}.self_attn.q_norm.weight"),
        &[4],
        vec![1.0; 4],
    );
    if layer < 2 {
        add_tensor(
            tensors,
            format!("{prefix}.self_attn.k_norm.weight"),
            &[4],
            vec![1.0; 4],
        );
    }
    add(tensors, "mlp.gate_proj.weight", &[12, HIDDEN], 18);
    add(tensors, "mlp.up_proj.weight", &[12, HIDDEN], 19);
    add(tensors, "mlp.down_proj.weight", &[HIDDEN, 12], 20);
    add(
        tensors,
        "per_layer_input_gate.weight",
        &[PLE_HIDDEN, HIDDEN],
        21,
    );
    add(
        tensors,
        "per_layer_projection.weight",
        &[HIDDEN, PLE_HIDDEN],
        22,
    );
    add_tensor(
        tensors,
        format!("{prefix}.post_per_layer_input_norm.weight"),
        &[HIDDEN],
        vec![1.0; HIDDEN],
    );
}

#[test]
#[ignore = "requires the pinned IREE runtime/compiler and a production target"]
fn token_and_dense_ple_match_with_mixed_cancel_and_slot_reuse() {
    let fixture = create_tiny_gemma3n();
    let tokens = [1, PER_LAYER_VOCAB as i32 + 1, 3];
    let prepared = fixture.prepared(&tokens);

    let mut token_session =
        XlaInferenceSession::load_with_context_capacity(fixture.path(), LAYERS, CAPACITY)
            .expect("token session");
    let mut prepared_session =
        XlaInferenceSession::load_with_context_capacity(fixture.path(), LAYERS, CAPACITY)
            .expect("PLE session");
    let token_output = token_session
        .generate_greedy(&tokens, 3, &[])
        .expect("token generation");
    let prepared_output = prepared_session
        .generate_gemma3n_prepared_greedy(&prepared, 3, &[])
        .expect("dense PLE generation");
    assert_eq!(prepared_output, token_output);

    // Full-capacity prefill drives shared-layer query RoPE through position
    // `2 * capacity - 1`, proving the extended table and no-truncation path.
    let near_capacity: Vec<i32> = (0..CAPACITY as i32)
        .map(|index| index % PER_LAYER_VOCAB as i32)
        .collect();
    let token_first = token_session
        .prefill_first_token(&near_capacity)
        .expect("near-capacity token prefill");
    let ple_first = prepared_session
        .prefill_gemma3n_prepared(&fixture.prepared(&near_capacity))
        .expect("near-capacity dense PLE prefill");
    assert_eq!(ple_first, token_first);

    let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string());
    let mut batch =
        XlaBatchEngine::load_with_context_capacity(fixture.path(), 4, &device, CAPACITY)
            .expect("Gemma3n batch");
    let cancelled = batch
        .submit_gemma3n_prepared(prepared.clone(), 3, SampleParams::greedy())
        .expect("cancellation candidate");
    assert!(batch.cancel(cancelled));
    assert!(batch.pump().expect("cancel pump").is_empty());

    let text = batch
        .submit(&tokens, 3, SampleParams::greedy())
        .expect("token request");
    let ple = batch
        .submit_gemma3n_prepared(prepared, 3, SampleParams::greedy())
        .expect("PLE request");
    let mut streams: HashMap<u64, Vec<i32>> = HashMap::new();
    while !batch.is_idle() {
        for event in batch.pump().expect("mixed Gemma3n pump") {
            if let EngineEvent::Token { req_id, token } = event {
                streams.entry(req_id).or_default().push(token);
            }
        }
    }
    assert_eq!(streams.get(&text), streams.get(&ple));
    assert_eq!(streams.get(&text).map(Vec::len), Some(3));

    let reused = batch
        .submit_gemma3n_prepared(fixture.prepared(&tokens), 2, SampleParams::greedy())
        .expect("reused PLE slot");
    assert!(
        batch
            .pump()
            .expect("reuse pump")
            .iter()
            .any(|event| matches!(event, EngineEvent::Token { req_id, .. } if *req_id == reused))
    );
}
