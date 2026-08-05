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

//! Unit tests for TeleChat3, concentrated on its RoPE.
//!
//! Everything here is checkpoint-free. The config test parses the real
//! `mlx-community/Telechat3-36B-Thinking-4bit` field set, and the model tests
//! build tiny synthetic weight maps whose key names mirror the checkpoint.
//!
//! The decoder itself is stock Llama and its shapes are unremarkable, so the
//! tests that carry weight are the RoPE ones. A dropped YaRN config leaves
//! every tensor the right shape and every short-prompt output plausible,
//! because YaRN and default RoPE agree closely at small offsets; the difference
//! only opens up past `original_max_position_embeddings`. Nothing downstream
//! can catch that, so it is pinned here.

use super::{ModelArgs, TeleChat3Model};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// Config surface.

/// The `mlx-community/Telechat3-36B-Thinking-4bit` config, field-for-field.
const TELECHAT3_36B_CONFIG: &str = r#"{
    "architectures": ["Telechat3ForCausalLM"],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "initializer_range": 0.0048,
    "intermediate_size": 24576,
    "max_position_embeddings": 32768,
    "mlp_bias": false,
    "model_type": "telechat3",
    "num_attention_heads": 48,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "pad_token_id": 3,
    "pretraining_tp": 1,
    "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "telechat3-yarn",
        "type": "telechat3-yarn"
    },
    "rope_theta": 1000000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.53.2",
    "use_cache": true,
    "vocab_size": 131072
}"#;

fn telechat3_36b() -> ModelArgs {
    serde_json::from_str(TELECHAT3_36B_CONFIG).expect("the real config must parse")
}

#[test]
fn the_real_config_parses() {
    let args = telechat3_36b();
    assert_eq!(args.model_type, "telechat3");
    assert_eq!(args.hidden_size, 6144);
    assert_eq!(args.num_hidden_layers, 64);
    assert_eq!(args.num_attention_heads, 48);
    assert_eq!(args.num_kv_heads(), 8);
    assert_eq!(args.intermediate_size, 24576);
    assert_eq!(args.vocab_size, 131_072);
    assert_eq!(args.rope_theta, 1_000_000.0);
    assert_eq!(args.max_position_embeddings, 32768);
    assert!(!args.tie_word_embeddings);
    assert_eq!(args.eos_token_ids(), vec![2]);

    // `head_dim` is explicit and is NOT hidden_size / num_attention_heads:
    // 6144 / 48 = 128 happens to agree here, but the field is authoritative.
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn the_published_checkpoint_sets_attention_bias_false() {
    // Worth pinning because the family is described as being distinguished by
    // attention_bias. The flag is plumbed, but it is off on the real
    // checkpoint, so it is not what makes this port different from stock Llama.
    let args = telechat3_36b();
    assert!(!args.attention_bias);
    assert!(!args.mlp_bias);
}

#[test]
fn the_telechat3_yarn_rope_type_actually_builds_a_yarn_table() {
    // The failure this guards: the shared YaRN reader accepts a fixed set of
    // `rope_type` spellings. If "telechat3-yarn" is not in that set the reader
    // returns None, the model silently falls back to a plain rotation at the
    // unscaled base, and every shape and every short-prompt output stays
    // plausible.
    let args = telechat3_36b();
    let yarn = args
        .yarn_rope()
        .expect("telechat3-yarn must produce a YaRN frequency table");

    assert_eq!(
        mlxcel_core::array_shape(&yarn.freqs),
        vec![64],
        "the table holds head_dim / 2 frequencies"
    );

    // yarn_get_mscale(4.0, 1) / yarn_get_mscale(4.0, 0) = (0.1 * ln 4 + 1) / 1.
    let expected = 0.1f32 * 4.0f32.ln() + 1.0;
    assert!(
        (yarn.mscale - expected).abs() < 1e-5,
        "mscale {} should be {expected}",
        yarn.mscale
    );
}

#[test]
fn the_vendor_prefix_is_an_alias_and_not_a_different_algorithm() {
    // Upstream dispatches "yarn", "deepseek_yarn" and "telechat3-yarn" onto the
    // same YarnRoPE. Same numbers under a plain "yarn" spelling must therefore
    // produce a bit-identical table.
    let args = telechat3_36b();
    let vendor = args.yarn_rope().expect("must build");

    let plain_cfg = TELECHAT3_36B_CONFIG.replace("telechat3-yarn", "yarn");
    let plain: ModelArgs = serde_json::from_str(&plain_cfg).expect("must parse");
    let plain = plain.yarn_rope().expect("must build");

    let diff = mlxcel_core::sum_all(&mlxcel_core::abs(&mlxcel_core::subtract(
        &vendor.freqs,
        &plain.freqs,
    )));
    mlxcel_core::eval(&diff);
    assert_eq!(mlxcel_core::item_f32(&diff), 0.0);
    assert_eq!(vendor.mscale, plain.mscale);
}

#[test]
fn the_rope_base_comes_from_the_config_not_the_readers_default() {
    // TeleChat3 keeps `rope_theta` at the top level while the shared reader
    // looks for it inside the scaling block and defaults to 500000 when it is
    // absent. Injecting the real 1000000 is what keeps the frequencies right;
    // a table built at 500000 is finite, correctly shaped and wrong.
    let args = telechat3_36b();
    let real = args.yarn_rope().expect("must build");

    let mut wrong_base = telechat3_36b();
    wrong_base.rope_theta = 500_000.0;
    let wrong = wrong_base.yarn_rope().expect("must build");

    let diff = mlxcel_core::sum_all(&mlxcel_core::abs(&mlxcel_core::subtract(
        &real.freqs,
        &wrong.freqs,
    )));
    mlxcel_core::eval(&diff);
    assert!(
        mlxcel_core::item_f32(&diff) > 1e-3,
        "a different rope_theta must produce a different frequency table, \
         otherwise the injection is not reaching the reader"
    );
}

#[test]
fn a_config_without_rope_scaling_uses_the_plain_rotation() {
    let cfg = r#"{
        "model_type": "telechat3",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rope_theta": 1000000.0
    }"#;
    let args: ModelArgs = serde_json::from_str(cfg).expect("must parse");
    assert!(args.yarn_rope().is_none());

    // An unknown rope_type is also a plain rotation rather than an error, which
    // matches the shared reader returning None for anything it does not know.
    let unknown = TELECHAT3_36B_CONFIG.replace("telechat3-yarn", "some-future-scheme");
    let args: ModelArgs = serde_json::from_str(&unknown).expect("must parse");
    assert!(args.yarn_rope().is_none());
}

// Model construction and forward.

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
        "model_type": "telechat3",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "telechat3-yarn"
        },
        "tie_word_embeddings": false,
        "eos_token_id": 2
    }"#,
    )
    .expect("the tiny config must parse")
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let ff = args.intermediate_size as i32;
    let vocab = args.vocab_size as i32;
    let q_out = (args.num_attention_heads * args.head_dim()) as i32;
    let kv_out = (args.num_kv_heads() * args.head_dim()) as i32;

    let mut w = WeightMap::new();
    w.insert("model.embed_tokens.weight".into(), filled(&[vocab, hidden]));
    w.insert("model.norm.weight".into(), ones(&[hidden]));
    w.insert("lm_head.weight".into(), filled(&[vocab, hidden]));

    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[hidden]));
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            ones(&[hidden]),
        );
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            filled(&[q_out, hidden]),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            filled(&[kv_out, hidden]),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            filled(&[kv_out, hidden]),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            filled(&[hidden, q_out]),
        );
        w.insert(format!("{p}.mlp.gate_proj.weight"), filled(&[ff, hidden]));
        w.insert(format!("{p}.mlp.up_proj.weight"), filled(&[ff, hidden]));
        w.insert(format!("{p}.mlp.down_proj.weight"), filled(&[hidden, ff]));
    }
    w
}

#[test]
fn from_weights_builds_the_stack_and_threads_yarn_into_every_layer() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = TeleChat3Model::from_weights(&weights, &args).expect("must build");

    assert_eq!(model.num_layers(), 2);
    assert!(model.lm_head.is_some());

    // Every layer has to carry the table, not just layer 0.
    for (i, layer) in model.layers.iter().enumerate() {
        assert!(
            layer.self_attn.rope_freqs.is_some(),
            "layer {i} lost its YaRN frequencies"
        );
    }
}

#[test]
fn forward_produces_vocab_sized_logits_and_advances_the_cache() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = TeleChat3Model::from_weights(&weights, &args).expect("must build");

    let mut caches = model.make_caches();
    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 4);

    let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
    let logits = LanguageModel::forward(&model, &next, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(caches[0].offset, 5);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, args.vocab_size as i32]
    );
}

#[test]
fn yarn_and_plain_rope_produce_different_logits() {
    // The end-to-end version of the RoPE guard: if the scaling were dropped
    // somewhere between config parsing and the rotation, these two models would
    // agree.
    let args = tiny_args();
    let weights = tiny_weights(&args);

    let mut plain = tiny_args();
    plain.rope_scaling = None;

    let with_yarn = TeleChat3Model::from_weights(&weights, &args).expect("must build");
    let without = TeleChat3Model::from_weights(&weights, &plain).expect("must build");

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6], &[1, 6]);

    let mut c1 = with_yarn.make_caches();
    let a = LanguageModel::forward(&with_yarn, &tokens, &mut c1, None);
    let mut c2 = without.make_caches();
    let b = LanguageModel::forward(&without, &tokens, &mut c2, None);

    let diff = mlxcel_core::sum_all(&mlxcel_core::abs(&mlxcel_core::subtract(&a, &b)));
    mlxcel_core::eval(&diff);
    assert!(
        mlxcel_core::item_f32(&diff) > 1e-3,
        "the YaRN table must reach the rotation"
    );
}

#[test]
fn a_missing_tensor_is_an_error() {
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    weights.remove("model.layers.1.self_attn.o_proj.weight");
    assert!(TeleChat3Model::from_weights(&weights, &args).is_err());
}

// Fixtures.

fn filled(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state: u32 = 0x9E37_79B9;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}
