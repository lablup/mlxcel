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

//! Equivalence gates between [`HeadlessLlama`] and the generator backbone.
//!
//! The embedding families that assemble a Llama decoder through
//! [`HeadlessLlama`] must reproduce [`Llama3Model::hidden_states`] bit for bit
//! under the same mask, for both entry points: token ids (the text-only
//! bidirectional families) and precomputed input embeddings (a vision family
//! that merges image features into the token embeddings first). These run on
//! a synthetic checkpoint-free model, so they hold on every backend.

use mlxcel_core::utils::create_bidirectional_padding_mask;
use mlxcel_core::weights::WeightMap;

use super::HeadlessLlama;
use crate::models::embedding_test_support::{Rng, ids_array, max_abs_diff, mlx_test_guard, to_vec};
use crate::models::llama3::{Llama3Model, ModelArgs};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 32;
const LAYERS: usize = 2;

fn tiny_args() -> ModelArgs {
    ModelArgs {
        model_type: "llama".to_string(),
        max_position_embeddings: None,
        hidden_size: HIDDEN,
        num_hidden_layers: LAYERS,
        intermediate_size: INTERMEDIATE,
        num_attention_heads: 2,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB,
        head_dim: Some(HEAD_DIM),
        num_key_value_heads: Some(1),
        attention_bias: false,
        mlp_bias: false,
        rope_theta: 500_000.0,
        rope_scaling: None,
        quantization: None,
        tie_word_embeddings: true,
        rope_traditional: false,
        eos_token_id: None,
        checkpoint_label: None,
    }
}

/// Deterministic dense weights in the generator's key layout.
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let mut rng = Rng::new(0x1424_0001);
    let mut w = WeightMap::new();
    let h = args.hidden_size as i32;
    let hd = args.head_dim() as i32;
    let q_out = args.num_attention_heads as i32 * hd;
    let kv_out = args.num_kv_heads() as i32 * hd;
    let inter = args.intermediate_size as i32;

    rng.insert(
        &mut w,
        "model.embed_tokens.weight",
        &[args.vocab_size as i32, h],
        0.5,
    );
    for i in 0..args.num_hidden_layers {
        let p = format!("model.layers.{i}");
        for (name, shape) in [
            ("self_attn.q_proj", [q_out, h]),
            ("self_attn.k_proj", [kv_out, h]),
            ("self_attn.v_proj", [kv_out, h]),
            ("self_attn.o_proj", [h, q_out]),
            ("mlp.gate_proj", [inter, h]),
            ("mlp.up_proj", [inter, h]),
            ("mlp.down_proj", [h, inter]),
        ] {
            rng.insert(&mut w, &format!("{p}.{name}.weight"), &shape, 0.2);
        }
        rng.insert(&mut w, &format!("{p}.input_layernorm.weight"), &[h], 0.1);
        rng.insert(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            0.1,
        );
    }
    rng.insert(&mut w, "model.norm.weight", &[h], 0.1);
    w
}

/// A right-padded two-row batch: the second row has three padding positions.
fn padded_batch() -> (Vec<i32>, Vec<i32>, i32, i32) {
    let length = 7;
    let ids = vec![1, 5, 9, 13, 17, 21, 25, 2, 6, 10, 14, 0, 0, 0];
    let mask = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0];
    (ids, mask, 2, length)
}

#[test]
fn token_id_path_matches_the_generator_bit_for_bit() {
    let _guard = mlx_test_guard();
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let headless = HeadlessLlama::from_weights(&weights, &args).expect("headless builds");
    let generator = Llama3Model::from_weights(&weights, &args).expect("generator builds");

    let (ids, mask, batch, length) = padded_batch();
    let input_ids = ids_array(&ids, batch, length);
    let bidirectional = create_bidirectional_padding_mask(&ids_array(&mask, batch, length));

    let mut caches = headless.make_caches();
    let ours =
        to_vec(&headless.forward_hidden(&input_ids, None, &mut caches, Some(&bidirectional)));
    let mut caches = generator.make_caches();
    let reference =
        to_vec(&generator.hidden_states(&input_ids, None, &mut caches, Some(&bidirectional)));

    assert_eq!(ours.len(), (batch * length) as usize * HIDDEN);
    assert_eq!(max_abs_diff(&ours, &reference), 0.0);
}

#[test]
fn input_embedding_path_matches_the_generator_bit_for_bit() {
    let _guard = mlx_test_guard();
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let headless = HeadlessLlama::from_weights(&weights, &args).expect("headless builds");
    let generator = Llama3Model::from_weights(&weights, &args).expect("generator builds");

    let (ids, mask, batch, length) = padded_batch();
    let input_ids = ids_array(&ids, batch, length);
    let bidirectional = create_bidirectional_padding_mask(&ids_array(&mask, batch, length));

    // The embedding tables are the same tensor, so both lookups agree; a vision
    // family would overwrite some rows with projected image features here.
    let ours_embeddings = headless.embed_tokens(&input_ids);
    let reference_embeddings = generator.get_embed_tokens(&input_ids);
    assert_eq!(
        max_abs_diff(&to_vec(&ours_embeddings), &to_vec(&reference_embeddings)),
        0.0
    );

    let mut caches = headless.make_caches();
    let ours = to_vec(&headless.forward_hidden(
        &input_ids,
        Some(&ours_embeddings),
        &mut caches,
        Some(&bidirectional),
    ));
    let mut caches = generator.make_caches();
    let reference = to_vec(&generator.hidden_states(
        &input_ids,
        Some(&reference_embeddings),
        &mut caches,
        Some(&bidirectional),
    ));

    assert_eq!(max_abs_diff(&ours, &reference), 0.0);
}

#[test]
fn a_missing_final_norm_names_the_key() {
    let _guard = mlx_test_guard();
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    weights.remove("model.norm.weight");
    let Err(error) = HeadlessLlama::from_weights(&weights, &args) else {
        panic!("a map without the final norm must not build");
    };
    assert!(error.contains("model.norm.weight"), "{error}");
}
