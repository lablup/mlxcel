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

//! Baichuan-M1 K/V convolution state must belong to one sequence (#2567).
//!
//! Every attention layer convolves the new K/V with the last pre-convolution
//! K/V token of the *same* sequence. The state therefore has to reset between
//! CLI generations and has to be keyed by `SequenceId` on the server, where
//! `--parallel 4` interleaves prefill and decode steps of different requests.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::{BaichuanConfig, BaichuanModel};
use crate::models::embedding_test_support::{Rng, max_abs_diff, mlx_test_guard, to_vec};

const HIDDEN: i32 = 16;
const INTERMEDIATE: i32 = 32;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const VOCAB: i32 = 32;
const LAYERS: usize = 2;

/// Two f32 layers; layer 1 is a sliding-window layer so both attention arms
/// of the family run.
fn tiny_config() -> BaichuanConfig {
    serde_json::from_value(json!({
        "model_type": "baichuan_m1",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rope_theta": 10000.0,
        "sliding_window": 4,
        "sliding_window_layers": [1],
        "conv_window": 2,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false
    }))
    .expect("tiny Baichuan config parses")
}

fn tiny_weights(config: &BaichuanConfig) -> WeightMap {
    let mut rng = Rng::new(0xBA1C_4AA7_2567);
    let mut w = WeightMap::new();
    let head_dim = HIDDEN / HEADS;
    let kv_out = KV_HEADS * head_dim;
    rng.insert(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN], 0.5);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.W_pack.weight"),
            &[HIDDEN + 2 * kv_out, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, HIDDEN],
            0.2,
        );
        // Convolution taps away from the identity so the previous token's
        // K/V visibly changes the current token's attention.
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.conv_k"),
            &[1, 1, KV_HEADS, 1, 2],
            0.9,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.conv_v"),
            &[1, 1, KV_HEADS, 1, 2],
            0.9,
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
    rng.insert(&mut w, "lm_head.weight", &[VOCAB, HIDDEN], 0.5);
    w
}

pub(crate) fn tiny_baichuan_model() -> BaichuanModel {
    let config = tiny_config();
    BaichuanModel::from_weights(&tiny_weights(&config), &config).expect("tiny Baichuan loads")
}

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// Prefill `prompt`, then decode `next` one token at a time, through the CLI
/// entry points (no sequence id). Returns the prefill logits followed by one
/// logits tensor per decode step.
fn generate_without_id(model: &BaichuanModel, prompt: &[i32], next: &[i32]) -> Vec<Vec<f32>> {
    let mut caches = LanguageModel::make_caches(model);
    let mut out = vec![to_vec(&LanguageModel::forward(
        model,
        &ids(prompt),
        &mut caches,
        None,
    ))];
    for &token in next {
        out.push(to_vec(&LanguageModel::forward(
            model,
            &ids(&[token]),
            &mut caches,
            None,
        )));
    }
    out
}

fn assert_steps_match(actual: &[Vec<f32>], expected: &[Vec<f32>], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (step, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = max_abs_diff(a, e);
        assert!(
            diff <= 1e-5,
            "{what}: step {step} differs from the sequence run alone (max abs diff {diff:e})"
        );
    }
}

/// Two generations of the same prompt on one model, each through
/// `make_caches`, must produce identical logits. Before #2567 the second
/// prompt was convolved with the last token of the first generation.
#[test]
fn consecutive_generations_through_make_caches_match() {
    let _guard = mlx_test_guard();
    let model = tiny_baichuan_model();
    let prompt = [1, 5, 9, 13, 17];
    let next = [21, 25, 29];

    let first = generate_without_id(&model, &prompt, &next);
    let second = generate_without_id(&model, &prompt, &next);

    for (step, (a, b)) in first.iter().zip(&second).enumerate() {
        let diff = max_abs_diff(a, b);
        assert!(
            diff == 0.0,
            "second generation step {step} inherited conv state from the first (max abs diff {diff:e})"
        );
    }
}

/// Prompts A and B of different lengths, prefilled and decoded under two
/// `SequenceId`s with alternating calls, plus one B = 2
/// `forward_batched_with_ids` decode step: every row must equal that prompt
/// generated alone.
#[test]
fn interleaved_sequences_match_each_sequence_alone() {
    let _guard = mlx_test_guard();
    let model = tiny_baichuan_model();
    let prompt_a = [2, 4, 6];
    let prompt_b = [3, 6, 9, 12, 15];
    let next_a = [8, 10, 12, 14];
    let next_b = [18, 21, 24, 27];

    let solo_a = generate_without_id(&model, &prompt_a, &next_a);
    let solo_b = generate_without_id(&model, &prompt_b, &next_b);

    let seq_a = SequenceId::from_raw(2567_01);
    let seq_b = SequenceId::from_raw(2567_02);
    LanguageModel::prepare_sequence_state(&model, seq_a);
    LanguageModel::prepare_sequence_state(&model, seq_b);
    let mut caches_a = LanguageModel::make_caches(&model);
    let mut caches_b = LanguageModel::make_caches(&model);

    let mut got_a = vec![to_vec(&model.forward_with_sequence_id(
        &ids(&prompt_a),
        Some(seq_a),
        &mut caches_a,
        None,
    ))];
    let mut got_b = vec![to_vec(&model.forward_with_sequence_id(
        &ids(&prompt_b),
        Some(seq_b),
        &mut caches_b,
        None,
    ))];
    for step in 0..3 {
        got_a.push(to_vec(&model.forward_with_sequence_id(
            &ids(&[next_a[step]]),
            Some(seq_a),
            &mut caches_a,
            None,
        )));
        got_b.push(to_vec(&model.forward_with_sequence_id(
            &ids(&[next_b[step]]),
            Some(seq_b),
            &mut caches_b,
            None,
        )));
    }

    // Fourth decode step as one two-row batch.
    let batched = {
        let input = mlxcel_core::from_slice_i32(&[next_a[3], next_b[3]], &[2, 1]);
        let mut rows: Vec<&mut [KVCache]> = vec![caches_a.as_mut_slice(), caches_b.as_mut_slice()];
        model.forward_batched_with_ids(&input, Some(&[seq_a, seq_b]), &mut rows, None)
    };
    mlxcel_core::eval(&batched);
    let shape = mlxcel_core::array_shape(&batched);
    assert_eq!(shape, vec![2, 1, VOCAB]);
    got_a.push(to_vec(&mlxcel_core::slice(
        &batched,
        &[0, 0, 0],
        &[1, 1, VOCAB],
    )));
    got_b.push(to_vec(&mlxcel_core::slice(
        &batched,
        &[1, 0, 0],
        &[2, 1, VOCAB],
    )));

    assert_steps_match(&got_a, &solo_a, "sequence A interleaved with B");
    assert_steps_match(&got_b, &solo_b, "sequence B interleaved with A");

    LanguageModel::release_sequence_state_by_id(&model, seq_a);
    LanguageModel::release_sequence_state_by_id(&model, seq_b);
}
