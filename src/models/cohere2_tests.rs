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

//! Unit tests for Cohere2 (Command R7B) on a tiny synthetic model.

use super::{Cohere2Config, Cohere2Model};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::sync::{Mutex, OnceLock};

/// Serialize the MLX-touching tests: MLX evaluation is not safe to run
/// concurrently from multiple test threads.
fn test_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

/// Four layers with `sliding_window_pattern = 2`, so both the sliding (RoPE)
/// and the global attention layers run, and a window shorter than the test
/// prompt, so the sliding prefill mask actually drops keys.
fn tiny_config() -> Cohere2Config {
    serde_json::from_str(
        r#"{
            "model_type": "cohere2",
            "hidden_size": 8,
            "head_dim": 4,
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10000.0,
            "vocab_size": 12,
            "layer_norm_eps": 1e-5,
            "logit_scale": 0.25,
            "sliding_window": 4,
            "sliding_window_pattern": 2,
            "use_embedding_sharing": true
        }"#,
    )
    .expect("parse tiny cohere2 config")
}

/// Deterministic small non-zero weights so activations stay bounded.
fn small(shape: &[i32], phase: usize) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase) as f32).sin() * 0.3)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn tiny_model() -> Cohere2Model {
    let cfg = tiny_config();
    let h = cfg.hidden_size as i32;
    let hd = cfg.head_dim as i32;
    let q_out = cfg.num_attention_heads as i32 * hd;
    let kv_out = cfg.num_key_value_heads as i32 * hd;
    let inter = cfg.intermediate_size as i32;

    let mut w = WeightMap::new();
    let mut phase = 1usize;
    let mut next = || {
        phase += 7;
        phase
    };
    w.insert(
        "model.embed_tokens.weight".into(),
        small(&[cfg.vocab_size as i32, h], next()),
    );
    w.insert("model.norm.weight".into(), ones(&[h]));
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        for (name, shape) in [
            ("self_attn.q_proj", [q_out, h]),
            ("self_attn.k_proj", [kv_out, h]),
            ("self_attn.v_proj", [kv_out, h]),
            ("self_attn.o_proj", [h, q_out]),
            ("mlp.gate_proj", [inter, h]),
            ("mlp.up_proj", [inter, h]),
            ("mlp.down_proj", [h, inter]),
        ] {
            w.insert(format!("{p}.{name}.weight"), small(&shape, next()));
        }
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[h]));
    }
    Cohere2Model::from_weights(&w, &cfg).expect("build tiny cohere2 model")
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let (a, b) = (array_to_vec_f32(a), array_to_vec_f32(b));
    assert_eq!(a.len(), b.len(), "shape mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Every last-logits entry point must return the full forward's row at
/// `last_pos`, `[1, 1, vocab]`, while projecting only that row through the LM
/// head. Checked at the final position (what prefill samples) and at an
/// interior one (what a padded prefill samples).
#[test]
fn last_logits_match_the_sliced_full_forward() {
    let _guard = test_guard().lock().unwrap();
    let model = tiny_model();
    let vocab = tiny_config().vocab_size as i32;
    let seq = [1i32, 3, 5, 7, 2, 9, 4];
    let ids = mlxcel_core::from_slice_i32(&seq, &[1, seq.len() as i32]);

    let mut caches = model.make_caches();
    let full = model.forward(&ids, &mut caches, None);
    mlxcel_core::eval(&full);

    for last_pos in [seq.len() - 1, 3] {
        let p = last_pos as i32;
        let expected = mlxcel_core::slice(&full, &[0, p, 0], &[1, p + 1, vocab]);

        let mut c1 = model.make_caches();
        let a = model.forward_last_logits(&ids, &mut c1, None, last_pos);
        let mut c2 = model.make_caches();
        let b = model.forward_last_logits_with_sequence_id(
            &ids,
            Some(SequenceId::from_raw(7)),
            &mut c2,
            None,
            last_pos,
        );
        let embeds = model
            .embed_tokens(&ids)
            .expect("cohere2 exposes embeddings");
        let mut c3 = model.make_caches();
        let c = model.forward_last_logits_with_embeddings_and_sequence_id(
            &ids,
            Some(&embeds),
            None,
            &mut c3,
            None,
            last_pos,
        );

        for (label, got) in [("plain", &a), ("sequence_id", &b), ("embeddings", &c)] {
            assert_eq!(
                mlxcel_core::array_shape(got),
                vec![1, 1, vocab],
                "{label} shape at {last_pos}"
            );
            let diff = max_abs_diff(&expected, got);
            assert!(diff < 1e-5, "{label} at {last_pos}: max |diff| {diff}");
        }
        // The last-logits path must still fill the caches for decode.
        assert_eq!(c1[0].offset, seq.len() as i32);
    }
}
