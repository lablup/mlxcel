// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::{Qwen2VLConfig, Qwen2VLModel};
use crate::models::embedding_test_support::{Rng, mlx_test_guard};
use crate::models::qwen_vl_batched_decode_test_support::RowPositionState;

const HIDDEN: i32 = 16;
const INTERMEDIATE: i32 = 24;
const HEAD_DIM: i32 = 8;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const VOCAB: i32 = 32;

pub(crate) fn tiny_qwen2_vl_model() -> Qwen2VLModel {
    let config: Qwen2VLConfig = serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": {"mrope_section": [2, 1, 1], "type": "default"},
        "tie_word_embeddings": true
    }))
    .expect("tiny Qwen2-VL config parses");
    let mut rng = Rng::new(0x2562_2002);
    let mut weights = WeightMap::new();
    rng.insert(
        &mut weights,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        0.4,
    );
    let p = "model.layers.0";
    for (name, shape) in [
        ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.o_proj.weight", vec![HIDDEN, HEADS * HEAD_DIM]),
        ("mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
        ("input_layernorm.weight", vec![HIDDEN]),
        ("post_attention_layernorm.weight", vec![HIDDEN]),
    ] {
        rng.insert(&mut weights, &format!("{p}.{name}"), &shape, 0.25);
    }
    rng.insert(&mut weights, "model.norm.weight", &[HIDDEN], 0.2);
    Qwen2VLModel::from_weights(&weights, &config).expect("tiny Qwen2-VL loads")
}

pub(crate) fn set_qwen2_state(
    model: &Qwen2VLModel,
    seq_id: SequenceId,
    state: RowPositionState,
    prompt_len: usize,
) {
    match state {
        RowPositionState::Text => {
            model.clear_mrope_state();
            model.bind_mrope_state_to_sequence(seq_id);
        }
        RowPositionState::Delta(delta) => {
            let positions = mlxcel_core::from_slice_i32(&[0, 0, 0], &[3, 1, 1]);
            model.set_mrope_state_for_sequence(seq_id, positions, delta);
        }
        RowPositionState::Stored { delta } => {
            let len = prompt_len + 1;
            let mut values = Vec::with_capacity(3 * len);
            for axis in 0..3 {
                values.extend((0..len).map(|position| position as i32 + axis * 17));
            }
            let positions = mlxcel_core::from_slice_i32(&values, &[3, 1, len as i32]);
            model.set_mrope_state_for_sequence(seq_id, positions, delta);
        }
    }
}

#[test]
fn qwen2_vl_native_batched_decode_covers_plain_mrope_scatter_and_fallbacks() {
    let _guard = mlx_test_guard();
    let model = tiny_qwen2_vl_model();
    crate::models::qwen_vl_batched_decode_test_support::assert_native_qwen_vl_contract(
        &model,
        |id, state, prompt_len| set_qwen2_state(&model, id, state, prompt_len),
    );
}
