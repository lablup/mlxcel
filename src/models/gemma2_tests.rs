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

//! Unit tests for Gemma 2 on a tiny synthetic model.

use super::{Gemma2Model, ModelArgs};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn filled(shape: &[i32], phase: usize) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase) as f32).sin() * 0.3)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// Two layers, tied embeddings (the `as_linear` head) and the default final
/// logit softcap.
fn tiny_model() -> Gemma2Model {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "gemma2",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "vocab_size": 12,
        "head_dim": 4,
        "tie_word_embeddings": true
    }))
    .expect("tiny gemma2 config parses");
    let (h, q, kv, ff) = (8, 8, 4, 16);
    let mut w = WeightMap::new();
    let mut phase = 1usize;
    let mut next = || {
        phase += 5;
        phase
    };
    w.insert("model.embed_tokens.weight".into(), filled(&[12, h], next()));
    w.insert("model.norm.weight".into(), filled(&[h], next()));
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", [q, h]),
            ("self_attn.k_proj.weight", [kv, h]),
            ("self_attn.v_proj.weight", [kv, h]),
            ("self_attn.o_proj.weight", [h, q]),
            ("mlp.gate_proj.weight", [ff, h]),
            ("mlp.up_proj.weight", [ff, h]),
            ("mlp.down_proj.weight", [h, ff]),
        ] {
            w.insert(format!("{p}.{name}"), filled(&shape, next()));
        }
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            w.insert(format!("{p}.{norm}.weight"), filled(&[h], next()));
        }
    }
    Gemma2Model::from_weights(&w, &args).expect("tiny gemma2 builds")
}

/// The last-logits overrides slice the hidden state before the LM head and
/// the softcap, and must return the full forward's row (#1968 follow-up).
#[test]
fn last_logits_match_the_sliced_full_forward() {
    let model = tiny_model();
    crate::test_support::last_logits::assert_last_logits_match_full_forward(
        &model,
        &[1, 3, 5, 7, 2, 9],
        true,
        1e-4,
    );
}
