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

//! Gate for the `forward_hidden` / `lm_head` split the Qwen3-VL-Embedding port
//! introduced.
//!
//! Generation goes through [`Qwen3VLModel::forward_for_sequence`], which is now
//! the head applied on top of [`Qwen3VLModel::forward_hidden`]. That has to be
//! a pure refactor, so this reassembles the head by hand and demands a
//! token-exact match against the public generation entry point.

use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::{Qwen3VLConfig, Qwen3VLModel};
use crate::models::embedding_test_support::{Rng, max_abs_diff, mlx_test_guard, to_vec};

const HIDDEN: i32 = 16;
const INTERMEDIATE: i32 = 32;
const HEAD_DIM: i32 = 8;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const VOCAB: i32 = 32;

/// A two-layer Qwen3-VL text config whose `mrope_section` sums to
/// `head_dim / 2`, as the published sections do.
fn tiny_config(num_layers: usize) -> Qwen3VLConfig {
    serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5_000_000.0,
        "rope_scaling": {"mrope_section": [2, 1, 1], "type": "default"},
        "tie_word_embeddings": true,
    }))
    .expect("tiny Qwen3-VL config parses")
}

/// Deterministic dense weights in the loader's post-remap key layout.
fn tiny_weights(config: &Qwen3VLConfig) -> WeightMap {
    let mut rng = Rng::new(0x0D15_EA5E_1345);
    let mut w = WeightMap::new();
    let q_out = HEADS * HEAD_DIM;
    let kv_out = KV_HEADS * HEAD_DIM;

    rng.insert(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN], 0.5);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[q_out, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv_out, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv_out, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, q_out],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            0.1,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            0.1,
        );
        rng.insert(
            &mut w,
            &format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.input_layernorm.weight"),
            &[HIDDEN],
            0.1,
        );
        rng.insert(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
            0.1,
        );
    }
    rng.insert(&mut w, "model.norm.weight", &[HIDDEN], 0.1);
    w
}

#[test]
fn forward_hidden_then_head_matches_forward_impl() {
    let _guard = mlx_test_guard();
    let config = tiny_config(2);
    let weights = tiny_weights(&config);
    let model = Qwen3VLModel::from_weights(&weights, &config).expect("synthetic Qwen3-VL loads");

    let ids: Vec<i32> = (0..12).map(|i| (i * 3 + 1) % VOCAB).collect();
    let input = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);

    let mut caches = model.make_caches();
    let logits = to_vec(&model.forward_impl(&input, None, &mut caches, None));

    let mut fresh = model.make_caches();
    let hidden = model.forward_hidden(&input, None, &mut fresh, None);
    let reassembled = to_vec(&model.lm_head.forward(&hidden));

    assert_eq!(logits.len(), reassembled.len());
    assert_eq!(
        max_abs_diff(&logits, &reassembled),
        0.0,
        "forward_hidden plus the head is not token-exact with forward_impl"
    );
}

#[test]
fn forward_hidden_returns_the_text_hidden_width() {
    let _guard = mlx_test_guard();
    let config = tiny_config(1);
    let model =
        Qwen3VLModel::from_weights(&tiny_weights(&config), &config).expect("synthetic loads");

    let ids: Vec<i32> = (0..5).collect();
    let input = mlxcel_core::from_slice_i32(&ids, &[1, 5]);
    let mut caches = model.make_caches();
    let hidden = model.forward_hidden(&input, None, &mut caches, None);
    assert_eq!(mlxcel_core::array_shape(&hidden), vec![1, 5, HIDDEN]);
}
