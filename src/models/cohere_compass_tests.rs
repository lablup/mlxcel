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

//! Gates for the pieces of the Compass decoder that a synthetic forward pass
//! cannot catch by accident: the `null`-means-no-RoPE reading of
//! `rope_parameters`, the split rotate-half layout, the parallel block, the
//! `logit_scale` multiply, the sliding/full window split, and the DeepStack
//! injection window. The MRoPE table itself is gated in
//! `cohere_compass_rope_tests.rs`.

use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::CohereCompassTextModel;
use crate::models::cohere_compass_config::CompassTextConfig;
use crate::models::embedding_test_support::{Rng, max_abs_diff, mlx_test_guard, to_vec};
use crate::models::qwen3_vl::apply_multimodal_rotary_pos_emb;

const HIDDEN: i32 = 16;
const INTERMEDIATE: i32 = 32;
const HEAD_DIM: i32 = 8;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const VOCAB: i32 = 32;

/// The published `rope_parameters` block, verbatim from
/// `North-Micro-Vision-Instruct`'s `text_config`.
fn published_rope_parameters() -> serde_json::Value {
    json!({
        "sliding_attention": {
            "mrope_interleaved": true,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
            "rope_theta": 50000
        },
        "full_attention": null,
        "rope_theta": 10000.0,
        "rope_type": "default"
    })
}

/// A tiny Compass config: `num_layers` layers, every 4th one full-attention,
/// with an `mrope_section` scaled down to `head_dim / 2 = 4` frequencies.
fn tiny_config(num_layers: usize) -> CompassTextConfig {
    let layer_types: Vec<&str> = (0..num_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(4) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "head_dim": HEAD_DIM,
        "layer_norm_eps": 1e-5,
        "logit_scale": 0.25,
        "sliding_window": 8,
        "layer_types": layer_types,
        "rope_parameters": {
            "sliding_attention": {
                "mrope_interleaved": true,
                "mrope_section": [2, 1, 1],
                "rope_type": "default",
                "rope_theta": 50000
            },
            "full_attention": null,
            "rope_theta": 10000.0,
            "rope_type": "default"
        },
        "tie_word_embeddings": true,
        "eos_token_id": 255001,
    }))
    .expect("tiny Compass config parses")
}

/// Deterministic dense weights in the loader's post-remap key layout. Note
/// what is absent: there is no `post_attention_layernorm` anywhere.
fn tiny_weights(config: &CompassTextConfig) -> WeightMap {
    let mut rng = Rng::new(0x0C0F_FEE0_1354);
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
    }
    rng.insert(&mut w, "model.norm.weight", &[HIDDEN], 0.1);
    w
}

#[test]
fn rope_parameters_null_means_no_rope() {
    let config: CompassTextConfig = serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 4,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "vocab_size": VOCAB,
        "rope_parameters": published_rope_parameters(),
    }))
    .expect("config parses");

    let sliding = config
        .rope_for_layer_type("sliding_attention")
        .expect("sliding entry resolves")
        .expect("sliding layers rotate");
    assert_eq!(sliding.theta, 50000.0);
    assert_eq!(sliding.mrope_section, [24, 20, 20]);

    assert!(
        config
            .rope_for_layer_type("full_attention")
            .expect("full entry resolves")
            .is_none(),
        "a JSON null rope_parameters entry must mean no positional encoding at all, not \
         'fall back to the default RoPE'"
    );
}

#[test]
fn rope_on_all_layers_applies_without_per_layer_entries() {
    let config: CompassTextConfig = serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 4,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "vocab_size": VOCAB,
        "rope_theta": 12345.0,
        "rope_on_all_layers": true,
    }))
    .expect("config parses");

    for layer_type in ["sliding_attention", "full_attention"] {
        let spec = config
            .rope_for_layer_type(layer_type)
            .expect("resolves")
            .unwrap_or_else(|| panic!("{layer_type} should rotate"));
        assert_eq!(spec.theta, 12345.0);
        assert_eq!(
            spec.mrope_section,
            crate::models::cohere_compass_rope::DEFAULT_MROPE_SECTION,
            "an entry with no mrope_section takes upstream's [22, 22, 20] default"
        );
    }
}

#[test]
fn rope_on_all_layers_false_disables_every_layer() {
    let config: CompassTextConfig = serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 4,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "vocab_size": VOCAB,
        "rope_on_all_layers": false,
    }))
    .expect("config parses");
    assert!(
        config
            .rope_for_layer_type("full_attention")
            .expect("resolves")
            .is_none()
    );
}

/// `rope_style: "split"` rotates the two halves of `head_dim`, not adjacent
/// pairs: with cos = 0 and sin = 1, `[1, 2, 3, 4]` must become `[-3, -4, 1, 2]`
/// and not the interleaved `[-2, 1, -4, 3]`.
#[test]
fn split_rope_layout() {
    let _guard = mlx_test_guard();
    let q = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
    let cos = mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4]);
    let sin = mlxcel_core::from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 4]);
    let (rotated, _) = apply_multimodal_rotary_pos_emb(&q, &q, &cos, &sin);
    assert_eq!(to_vec(&rotated), vec![-3.0, -4.0, 1.0, 2.0]);
}

/// The block is parallel: attention and MLP both read the *same* normed input
/// and both add onto the residual. A synthetic checkpoint that also carries a
/// `post_attention_layernorm` must load and produce the identical result, since
/// this family has no such norm.
#[test]
fn parallel_block_has_no_post_norm() {
    let _guard = mlx_test_guard();
    let config = tiny_config(2);
    let weights = tiny_weights(&config);

    let mut with_stray = WeightMap::new();
    for (k, v) in &weights {
        with_stray.insert(k.clone(), mlxcel_core::copy(v));
    }
    let mut rng = Rng::new(0xDEAD_BEEF);
    for i in 0..config.num_hidden_layers {
        rng.insert(
            &mut with_stray,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
            &[HIDDEN],
            0.7,
        );
    }

    let baseline = CohereCompassTextModel::from_weights(&weights, &config).expect("baseline loads");
    let stray = CohereCompassTextModel::from_weights(&with_stray, &config)
        .expect("a stray post_attention_layernorm must not break loading");

    let ids: Vec<i32> = (0..9).map(|i| (i * 5 + 1) % VOCAB).collect();
    let input = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);

    let mut c1 = baseline.make_caches();
    let mut c2 = stray.make_caches();
    let a = to_vec(&baseline.forward_impl(&input, None, &mut c1));
    let b = to_vec(&stray.forward_impl(&input, None, &mut c2));
    assert_eq!(
        max_abs_diff(&a, &b),
        0.0,
        "the parallel block must ignore a post_attention_layernorm key"
    );
}

/// `logit_scale` multiplies the head output. Two models that differ only in
/// `logit_scale` must differ by exactly that ratio.
#[test]
fn logit_scale_applied() {
    let _guard = mlx_test_guard();
    let mut scaled = tiny_config(2);
    scaled.logit_scale = 0.25;
    let mut unscaled = tiny_config(2);
    unscaled.logit_scale = 1.0;
    let weights = tiny_weights(&scaled);

    let m_scaled = CohereCompassTextModel::from_weights(&weights, &scaled).expect("loads");
    let m_unscaled = CohereCompassTextModel::from_weights(&weights, &unscaled).expect("loads");

    let ids: Vec<i32> = (0..6).collect();
    let input = mlxcel_core::from_slice_i32(&ids, &[1, 6]);
    let mut c1 = m_scaled.make_caches();
    let mut c2 = m_unscaled.make_caches();
    let a = to_vec(&m_scaled.forward_impl(&input, None, &mut c1));
    let b = to_vec(&m_unscaled.forward_impl(&input, None, &mut c2));

    let quarter: Vec<f32> = b.iter().map(|v| v * 0.25).collect();
    assert!(
        max_abs_diff(&a, &quarter) < 1e-4,
        "logit_scale is not applied after the (tied) head"
    );
}

/// 28 layers, sliding everywhere except `{3, 7, 11, 15, 19, 23, 27}`. The repo
/// enforces the window through the prefill mask plus the SDPA `window_size`
/// over a dense `KVCache` (`cohere2.rs` / `olmo3.rs`), not a rotating buffer,
/// so this asserts the per-layer window split rather than a cache type.
#[test]
fn sliding_layers_carry_the_window_and_full_layers_do_not() {
    let _guard = mlx_test_guard();
    let config = tiny_config(28);
    let model = CohereCompassTextModel::from_weights(&tiny_weights(&config), &config)
        .expect("28-layer synthetic loads");

    assert_eq!(model.num_layers(), 28);
    assert_eq!(model.make_caches().len(), 28);
    let full: Vec<usize> = model
        .layer_is_sliding()
        .iter()
        .enumerate()
        .filter_map(|(i, &sliding)| (!sliding).then_some(i))
        .collect();
    assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27]);
    assert_eq!(model.sliding_window(), 8);
}

/// A `mask == None` prefill must still be causal on both layer types: the
/// logits of the first 48 rows of a 96-token prefill have to match a 48-token
/// prefill of the same prefix.
#[test]
fn text_prefill_is_causal() {
    let _guard = mlx_test_guard();
    // 4 layers: 3 sliding + 1 full, with a window wide enough that the
    // sliding layers see the whole 96-token prefix.
    let mut config = tiny_config(4);
    config.sliding_window = 128;
    let model = CohereCompassTextModel::from_weights(&tiny_weights(&config), &config)
        .expect("4-layer synthetic loads");

    let ids: Vec<i32> = (0..96).map(|i| (i * 7 + 3) % VOCAB).collect();
    let long = mlxcel_core::from_slice_i32(&ids, &[1, 96]);
    let short = mlxcel_core::from_slice_i32(&ids[..48], &[1, 48]);

    let mut c_long = model.make_caches();
    let mut c_short = model.make_caches();
    let long_logits = to_vec(&model.forward_impl(&long, None, &mut c_long));
    let short_logits = to_vec(&model.forward_impl(&short, None, &mut c_short));

    let width = VOCAB as usize;
    let prefix = &long_logits[..48 * width];
    assert_eq!(prefix.len(), short_logits.len());
    assert!(
        max_abs_diff(prefix, &short_logits) < 1e-4,
        "a maskless prefill leaked future tokens into the first 48 positions"
    );
}

/// DeepStack features are injected after the first `k` layers only, where `k`
/// is the number of branch outputs the vision tower produced (3 for the
/// published checkpoint). Injecting into a fourth layer would change the
/// result; not injecting at all would leave it equal to the no-DeepStack run.
#[test]
fn deepstack_injects_first_three_layers_only() {
    let _guard = mlx_test_guard();
    let config = tiny_config(6);
    let model = CohereCompassTextModel::from_weights(&tiny_weights(&config), &config)
        .expect("6-layer synthetic loads");

    let ids: Vec<i32> = (0..8).map(|i| (i * 3 + 2) % VOCAB).collect();
    let input = mlxcel_core::from_slice_i32(&ids, &[1, 8]);
    let embeds = model.get_embed_tokens(&input);

    // Positions 2..=4 are the "image" run.
    let mask_vals = [0.0f32, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let mask = mlxcel_core::from_slice_f32(&mask_vals, &[1, 8]);
    let mask = mlxcel_core::astype(&mask, mlxcel_core::dtype::BOOL);

    let mut caches = model.make_caches();
    let plain = to_vec(&model.forward_impl(&input, Some(&embeds), &mut caches));

    let three: Vec<_> = (0..3)
        .map(|k| {
            mlxcel_core::full_f32(
                &[3, HIDDEN],
                0.1 * (k as f32 + 1.0),
                mlxcel_core::dtype::FLOAT32,
            )
        })
        .collect();
    model.set_deepstack_state(mlxcel_core::copy(&mask), three);
    let mut caches = model.make_caches();
    let injected3 = to_vec(&model.forward_impl(&input, Some(&embeds), &mut caches));
    model.clear_deepstack_state();

    let four: Vec<_> = (0..4)
        .map(|k| {
            mlxcel_core::full_f32(
                &[3, HIDDEN],
                0.1 * (k as f32 + 1.0),
                mlxcel_core::dtype::FLOAT32,
            )
        })
        .collect();
    model.set_deepstack_state(mlxcel_core::copy(&mask), four);
    let mut caches = model.make_caches();
    let injected4 = to_vec(&model.forward_impl(&input, Some(&embeds), &mut caches));
    model.clear_deepstack_state();

    assert!(
        max_abs_diff(&plain, &injected3) > 1e-5,
        "three DeepStack branches must change the logits"
    );
    assert!(
        max_abs_diff(&injected3, &injected4) > 1e-5,
        "a fourth DeepStack branch must reach layer 3; the injection window follows the branch \
         count, not a hard-coded 3"
    );
}

/// A `sequential` block type is refused rather than silently run as parallel.
#[test]
fn sequential_block_type_is_rejected() {
    let mut config = tiny_config(2);
    config.transformer_block_type = "sequential".to_string();
    let err = match CohereCompassTextModel::from_weights(&tiny_weights(&config), &config) {
        Ok(_) => panic!("sequential must not silently load as parallel"),
        Err(e) => e,
    };
    assert!(err.contains("transformer_block_type"), "{err}");
}
