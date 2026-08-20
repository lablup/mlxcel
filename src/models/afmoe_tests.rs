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

//! Unit tests for the Arcee AFMoE loader.
//!
//! Everything here is checkpoint-free. The groups that carry the most weight
//! cover what a passing real-checkpoint run cannot distinguish from a wrong
//! implementation:
//!
//! 1. **The hybrid schedule comes from `layer_types`, not a modulus.** The
//!    config declares `global_attn_every_n_layers` beside the list and upstream
//!    never reads it, so a port that trusts the modulus agrees with Trinity by
//!    coincidence and diverges on any checkpoint whose list is irregular.
//! 2. **The muP embedding scale**, which is a config flag rather than a tensor,
//!    so nothing in the checkpoint disagrees if it is dropped.
//! 3. **The four sandwich norms and the attention gate**, the tensors a
//!    qwen3_moe-shaped loader would never look for.
//! 4. **Reference-free causality**, since generation calls `forward` with
//!    `mask == None` and a bidirectional prefill is fluent and wrong.

use super::{AfmoeModel, ModelArgs, Quantization, ScoreFunc, validate_weights};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// The real checkpoint's config.

/// `mlx-community/Trinity-Nano-Preview-4bit`'s `config.json`.
///
/// `layer_types` is truncated to its first 8 entries (the real file lists 56)
/// and `num_hidden_layers` set to match, so the constant stays readable.
/// Everything else is field for field, including keys this loader ignores.
const TRINITY_CONFIG: &str = r#"{
    "architectures": ["AfmoeForCausalLM"],
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "global_attn_every_n_layers": 4,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
    ],
    "load_balance_coeff": 0.001,
    "max_position_embeddings": 131072,
    "model_type": "afmoe",
    "moe_intermediate_size": 256,
    "mup_enabled": true,
    "n_group": 1,
    "num_attention_heads": 8,
    "num_dense_layers": 2,
    "num_expert_groups": 1,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 8,
    "num_key_value_heads": 2,
    "num_limited_groups": 1,
    "num_shared_experts": 1,
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000,
    "route_norm": true,
    "route_scale": 2.826,
    "score_func": "sigmoid",
    "sliding_window": 2048,
    "tie_word_embeddings": false,
    "topk_group": 1,
    "use_cache": true,
    "use_grouped_mm": true,
    "vocab_size": 200192
}"#;

fn trinity_args() -> ModelArgs {
    serde_json::from_str(TRINITY_CONFIG).expect("Trinity config parses")
}

// Config parsing.

#[test]
fn the_real_config_parses_and_validates() {
    let args = trinity_args();
    assert_eq!(args.model_type, "afmoe");
    assert_eq!(args.hidden_size, 1024);
    assert_eq!(args.head_dim, 128);
    assert_eq!(args.num_attention_heads, 8);
    assert_eq!(args.num_key_value_heads, 2);
    assert_eq!(args.intermediate_size, 3072);
    assert_eq!(args.moe_intermediate_size, 256);
    assert_eq!(args.num_experts, 128);
    assert_eq!(args.num_experts_per_tok, 8);
    assert_eq!(args.num_shared_experts, 1);
    assert_eq!(args.num_dense_layers, 2);
    assert_eq!(args.sliding_window, 2048);
    assert_eq!(args.route_scale, 2.826);
    assert!(args.route_norm);
    assert!(args.mup_enabled);
    assert_eq!(args.score_func(), ScoreFunc::Sigmoid);
    assert_eq!(args.n_group, 1);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    args.validate().expect("the shipped config is accepted");
}

#[test]
fn the_head_dim_is_read_and_not_derived() {
    // AFMoE declares head_dim rather than deriving hidden_size / num_heads. On
    // Trinity-Nano the two coincide at 128, which is exactly the coincidence
    // that would hide a derived implementation until a sibling breaks it.
    let args = trinity_args();
    assert_eq!(args.hidden_size / args.num_attention_heads, 128);
    assert_eq!(args.head_dim, 128);

    let mut args = trinity_args();
    args.head_dim = 64;
    assert_eq!(args.head_dim, 64, "the declared width wins over the ratio");
    args.validate()
        .expect("a head width that disagrees with the ratio is legal");
}

// The hybrid schedule.

#[test]
fn the_schedule_comes_from_layer_types_not_the_modulus() {
    let args = trinity_args();
    let sliding: Vec<bool> = (0..args.num_hidden_layers)
        .map(|i| args.is_sliding_layer(i))
        .collect();
    assert_eq!(
        sliding,
        vec![true, true, true, false, true, true, true, false]
    );
    assert_eq!(args.full_attention_index(), Some(3));
    assert_eq!(args.sliding_index(), Some(0));
}

#[test]
fn an_irregular_layer_types_list_is_honoured_over_global_attn_every_n_layers() {
    // `global_attn_every_n_layers` is declared in the config and upstream never
    // reads it. A port that trusted the modulus would agree with the list on
    // Trinity and disagree here, which is a case no real-checkpoint run reaches.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "afmoe", "num_hidden_layers": 5, "global_attn_every_n_layers": 4,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                            "sliding_attention", "sliding_attention"]}"#,
    )
    .expect("config parses");
    let sliding: Vec<bool> = (0..5).map(|i| args.is_sliding_layer(i)).collect();
    assert_eq!(sliding, vec![false, true, true, true, true]);
    // The modulus would have made layer 3 global; the list says otherwise.
    assert!(args.is_sliding_layer(3));
    assert_eq!(args.full_attention_index(), Some(0));
}

#[test]
fn a_layer_types_list_shorter_than_the_stack_is_rejected() {
    let mut args = trinity_args();
    args.num_hidden_layers = 12; // the list only covers 8
    let err = args
        .validate()
        .expect_err("a short schedule would silently make the tail global");
    assert!(err.contains("layer_types"), "{err}");
}

#[test]
fn an_unrecognized_layer_type_is_rejected() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "afmoe", "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "chunked_attention"]}"#,
    )
    .expect("config parses");
    let err = args.validate().expect_err("only two kinds are defined");
    assert!(err.contains("chunked_attention"), "{err}");
}

// The muP scale.

#[test]
fn the_mup_scale_is_the_square_root_of_hidden_size() {
    let args = trinity_args();
    assert!((args.embedding_scale() - 32.0).abs() < 1e-6, "sqrt(1024)");

    let mut args = trinity_args();
    args.mup_enabled = false;
    assert_eq!(args.embedding_scale(), 1.0);
}

// Config guards.

#[test]
fn a_top_k_larger_than_the_expert_count_is_rejected() {
    let mut args = trinity_args();
    args.num_experts_per_tok = 129;
    let err = args
        .validate()
        .expect_err("129 of 128 experts is out of range");
    assert!(err.contains("num_experts_per_tok"), "{err}");
}

#[test]
fn the_grouped_routing_guards_fire_even_though_trinity_never_reaches_them() {
    // n_group is 1 on every published AFMoE checkpoint, so this whole branch is
    // unreachable in a real-model run and is pinned here instead.
    let mut args = trinity_args();
    args.n_group = 4;
    args.topk_group = 4;
    let err = args
        .validate()
        .expect_err("topk_group == n_group puts argpartition out of range");
    assert!(err.contains("topk_group"), "{err}");

    let mut args = trinity_args();
    args.n_group = 5; // 128 is not divisible by 5
    args.topk_group = 2;
    let err = args
        .validate()
        .expect_err("an indivisible expert count cannot be regrouped");
    assert!(err.contains("n_group"), "{err}");

    let mut args = trinity_args();
    args.n_group = 128; // one expert per group
    args.topk_group = 2;
    let err = args
        .validate()
        .expect_err("a group of one cannot be scored by its top two");
    assert!(err.contains("per group"), "{err}");
}

#[test]
fn an_unrecognized_score_func_is_rejected() {
    let mut args = trinity_args();
    args.score_func = "relu".into();
    let err = args.validate().expect_err("only sigmoid and softmax exist");
    assert!(err.contains("score_func"), "{err}");
}

#[test]
fn a_zero_sliding_window_is_rejected_when_a_layer_slides() {
    let mut args = trinity_args();
    args.sliding_window = 0;
    let err = args
        .validate()
        .expect_err("a zero window would keep no keys");
    assert!(err.contains("sliding_window"), "{err}");

    // With no sliding layer the window is never used, so it is not an error.
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "afmoe", "num_hidden_layers": 2, "sliding_window": 0,
            "layer_types": ["full_attention", "full_attention"], "num_dense_layers": 2}"#,
    )
    .expect("config parses");
    args.validate().expect("an unused window may be anything");
}

#[test]
fn an_odd_head_dim_is_rejected() {
    let mut args = trinity_args();
    args.head_dim = 127;
    let err = args.validate().expect_err("RoPE rotates channel pairs");
    assert!(err.contains("head_dim"), "{err}");
}

#[test]
fn a_non_default_rope_scaling_block_is_rejected() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type": "afmoe", "num_hidden_layers": 1, "num_dense_layers": 1,
            "layer_types": ["full_attention"],
            "rope_scaling": {"rope_type": "linear", "factor": 4.0}}"#,
    )
    .expect("config parses");
    let err = args
        .validate()
        .expect_err("a scaled rope block is rejected");
    assert!(err.contains("rope_scaling"), "{err}");
}

#[test]
fn a_zero_scalar_is_rejected_before_it_divides() {
    for mutate in [
        (|a: &mut ModelArgs| a.num_attention_heads = 0) as fn(&mut ModelArgs),
        |a: &mut ModelArgs| a.hidden_size = 0,
        |a: &mut ModelArgs| a.head_dim = 0,
        |a: &mut ModelArgs| a.num_hidden_layers = 0,
        |a: &mut ModelArgs| a.num_key_value_heads = 0,
        |a: &mut ModelArgs| a.vocab_size = 0,
        |a: &mut ModelArgs| a.intermediate_size = 0,
        |a: &mut ModelArgs| a.moe_intermediate_size = 0,
        |a: &mut ModelArgs| a.rms_norm_eps = 0.0,
        |a: &mut ModelArgs| a.rope_theta = 0.0,
    ] {
        let mut args = trinity_args();
        mutate(&mut args);
        assert!(
            args.validate().is_err(),
            "a zero scalar must be rejected at load"
        );
    }
}

// Weight-shape validation.

fn lazy(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(shape, 0.0, mlxcel_core::dtype::FLOAT32)
}

fn synthetic_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let head_dim = args.head_dim as i32;
    let q_size = args.num_attention_heads as i32 * head_dim;
    let kv_size = args.num_key_value_heads as i32 * head_dim;
    let experts = args.num_experts as i32;
    let moe = args.moe_intermediate_size as i32;
    let mut weights = WeightMap::new();

    weights.insert(
        "model.embed_tokens.weight".into(),
        lazy(&[args.vocab_size as i32, hidden]),
    );
    weights.insert("model.norm.weight".into(), lazy(&[hidden]));
    weights.insert(
        "lm_head.weight".into(),
        lazy(&[args.vocab_size as i32, hidden]),
    );

    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let attn = format!("{prefix}.self_attn");
        weights.insert(format!("{attn}.q_proj.weight"), lazy(&[q_size, hidden]));
        weights.insert(format!("{attn}.k_proj.weight"), lazy(&[kv_size, hidden]));
        weights.insert(format!("{attn}.v_proj.weight"), lazy(&[kv_size, hidden]));
        weights.insert(format!("{attn}.o_proj.weight"), lazy(&[hidden, q_size]));
        weights.insert(format!("{attn}.gate_proj.weight"), lazy(&[q_size, hidden]));
        weights.insert(format!("{attn}.q_norm.weight"), lazy(&[head_dim]));
        weights.insert(format!("{attn}.k_norm.weight"), lazy(&[head_dim]));
        for leaf in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "post_mlp_layernorm",
        ] {
            weights.insert(format!("{prefix}.{leaf}.weight"), lazy(&[hidden]));
        }

        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            weights.insert(
                format!("{mlp}.router.gate.weight"),
                lazy(&[experts, hidden]),
            );
            weights.insert(format!("{mlp}.expert_bias"), lazy(&[experts]));
            weights.insert(
                format!("{mlp}.experts.gate_proj.weight"),
                lazy(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.experts.up_proj.weight"),
                lazy(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.experts.down_proj.weight"),
                lazy(&[experts, hidden, moe]),
            );
            let shared = args.shared_expert_intermediate_size() as i32;
            weights.insert(
                format!("{mlp}.shared_experts.gate_proj.weight"),
                lazy(&[shared, hidden]),
            );
            weights.insert(
                format!("{mlp}.shared_experts.up_proj.weight"),
                lazy(&[shared, hidden]),
            );
            weights.insert(
                format!("{mlp}.shared_experts.down_proj.weight"),
                lazy(&[hidden, shared]),
            );
        } else {
            let inter = args.intermediate_size as i32;
            weights.insert(format!("{mlp}.gate_proj.weight"), lazy(&[inter, hidden]));
            weights.insert(format!("{mlp}.up_proj.weight"), lazy(&[inter, hidden]));
            weights.insert(format!("{mlp}.down_proj.weight"), lazy(&[hidden, inter]));
        }
    }
    weights
}

/// A shrunk Trinity: 6 layers (2 dense), 4 heads of 16, 8 experts.
fn small_args() -> ModelArgs {
    let mut args = trinity_args();
    args.hidden_size = 64;
    args.head_dim = 16;
    args.num_attention_heads = 4;
    args.num_key_value_heads = 2;
    args.num_hidden_layers = 6;
    args.layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    args.intermediate_size = 128;
    args.moe_intermediate_size = 32;
    args.num_experts = 8;
    args.num_experts_per_tok = 2;
    args.num_dense_layers = 2;
    args.vocab_size = 96;
    args.sliding_window = 8;
    args.quantization = Some(Quantization {
        group_size: 64,
        bits: 4,
    });
    args
}

#[test]
fn a_well_formed_checkpoint_passes_validation() {
    let args = small_args();
    args.validate().expect("the shrunk config is valid");
    let weights = synthetic_weights(&args);
    validate_weights(&weights, &args).expect("the synthetic export validates");
}

#[test]
fn a_missing_attention_gate_is_rejected() {
    // `self_attn.gate_proj` is the feature qwen3_moe has no counterpart for, so
    // a loader modelled on that family would never look for it.
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.remove("model.layers.0.self_attn.gate_proj.weight");
    let err = validate_weights(&weights, &args).expect_err("the attention gate is required");
    assert!(err.contains("gate_proj"), "{err}");
}

#[test]
fn all_four_norms_are_required() {
    // A two-norm pre-norm block would load every other tensor and still
    // generate, so each of the four is checked by name.
    for leaf in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_mlp_layernorm",
    ] {
        let args = small_args();
        let mut weights = synthetic_weights(&args);
        weights.remove(&format!("model.layers.0.{leaf}.weight"));
        match validate_weights(&weights, &args) {
            Ok(()) => panic!("{leaf} must be required"),
            Err(err) => assert!(err.contains(leaf), "{err}"),
        }
    }
}

#[test]
fn a_short_expert_stack_is_rejected() {
    let args = small_args();
    let mut weights = synthetic_weights(&args);
    weights.insert(
        "model.layers.2.mlp.experts.gate_proj.weight".into(),
        lazy(&[
            args.num_experts as i32 - 1,
            args.moe_intermediate_size as i32,
            args.hidden_size as i32,
        ]),
    );
    let err = validate_weights(&weights, &args).expect_err("a short stack is rejected");
    assert!(err.contains("num_experts"), "{err}");
}

#[test]
fn a_dense_prefix_layer_is_validated_as_a_plain_mlp() {
    let args = small_args();
    let weights = synthetic_weights(&args);
    assert!(!args.is_moe_layer(0) && !args.is_moe_layer(1));
    assert!(args.is_moe_layer(2));
    assert!(weights.contains_key("model.layers.0.mlp.gate_proj.weight"));
    assert!(!weights.contains_key("model.layers.0.mlp.router.gate.weight"));
    validate_weights(&weights, &args).expect("the dense prefix validates");
}

// End-to-end construction and forward.

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn filled_weights(args: &ModelArgs) -> WeightMap {
    let mut weights = synthetic_weights(args);
    let mut keys: Vec<String> = weights.keys().cloned().collect();
    // Sorted because `WeightMap` is a `HashMap`: its iteration order is
    // randomized per process, and the seed below advances once per key, so an
    // unsorted walk builds a different random model on every run (issue #1265).
    keys.sort();
    let mut seed = 0xA1B2_C3D4u32;
    for key in keys {
        let shape = mlxcel_core::array_shape(weights.get(&key).expect("key just listed"));
        let n: i32 = shape.iter().product();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        weights.insert(
            key,
            mlxcel_core::from_slice_f32(&noise(n as usize, seed), &shape),
        );
    }
    weights
}

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::reshape(array, &[-1]);
    let n = mlxcel_core::array_shape(&flat)[0];
    (0..n)
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(&flat, &[i], &[i + 1])))
        .collect()
}

#[test]
fn a_synthetic_model_builds_and_produces_finite_logits() {
    let args = small_args();
    let weights = filled_weights(&args);
    let model = AfmoeModel::from_weights(&weights, &args).expect("the model builds");
    assert_eq!(model.num_layers(), args.num_hidden_layers);

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5], &[1, 5]);
    let mut caches = LanguageModel::make_caches(&model);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 5, args.vocab_size as i32]
    );
    for (i, value) in read_all(&mlxcel_core::slice(&logits, &[0, 4, 0], &[1, 5, 16]))
        .iter()
        .enumerate()
    {
        assert!(value.is_finite(), "logit[{i}] is {value}");
    }

    let next = mlxcel_core::from_slice_i32(&[6], &[1, 1]);
    let step = LanguageModel::forward(&model, &next, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&step),
        vec![1, 1, args.vocab_size as i32]
    );
}

#[test]
fn the_prefill_is_causal_without_being_handed_a_mask() {
    // Generation calls `forward` with `mask == None` and the model builds its
    // own masks, one causal and one windowed. A bidirectional prefill produces
    // fluent text, so position 0 is compared directly against a one-token run.
    let args = small_args();
    let weights = filled_weights(&args);
    let model = AfmoeModel::from_weights(&weights, &args).expect("the model builds");
    let vocab = args.vocab_size as i32;

    let one = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
    let mut caches = LanguageModel::make_caches(&model);
    let single = LanguageModel::forward(&model, &one, &mut caches, None);
    let single_row = read_all(&mlxcel_core::slice(&single, &[0, 0, 0], &[1, 1, vocab]));

    let many = mlxcel_core::from_slice_i32(&[7, 11, 13, 17], &[1, 4]);
    let mut caches = LanguageModel::make_caches(&model);
    let prefill = LanguageModel::forward(&model, &many, &mut caches, None);
    let first_row = read_all(&mlxcel_core::slice(&prefill, &[0, 0, 0], &[1, 1, vocab]));

    for (i, (a, b)) in single_row.iter().zip(first_row.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "logit {i} at position 0 changed when later tokens were added: {a} vs {b}. \
             The prefill is not causal."
        );
    }
}

/// A cache that reports whatever offset the test asks for and returns fixed
/// keys and values, so an attention block can be run at two different sequence
/// positions over identical inputs.
struct StubCache {
    offset: i32,
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
}

impl crate::models::gemma3::CacheInterface for StubCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn live_len(&self) -> i32 {
        mlxcel_core::array_shape(&self.keys)[2]
    }

    fn update_and_fetch(
        &mut self,
        _k: UniquePtr<MlxArray>,
        _v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (
            mlxcel_core::copy(&self.keys),
            mlxcel_core::copy(&self.values),
        )
    }
}

#[test]
fn a_full_attention_layer_is_nope_and_a_sliding_one_is_not() {
    // Upstream builds `self.rope` only when `is_local_attention`, so global
    // layers apply no positional encoding at all. Rotating them anyway loads
    // every tensor, costs nothing, and is invisible at position 0 (where the
    // rotation is the identity), so it is checked directly: a NoPE block must
    // return the same thing at offset 0 and offset 1000 for identical inputs,
    // and a rotating one must not.
    use super::Attention;
    use crate::models::gemma3::CacheInterface;

    let args = small_args();
    let hidden = args.hidden_size as i32;
    let head_dim = args.head_dim as i32;
    let q_size = args.num_attention_heads as i32 * head_dim;
    let kv_size = args.num_key_value_heads as i32 * head_dim;

    let mut weights = WeightMap::new();
    let mut seed = 0x0BAD_F00Du32;
    let put = |weights: &mut WeightMap, key: &str, shape: &[i32], seed: &mut u32| {
        let n: i32 = shape.iter().product();
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        weights.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&noise(n as usize, *seed), shape),
        );
    };
    put(
        &mut weights,
        "a.q_proj.weight",
        &[q_size, hidden],
        &mut seed,
    );
    put(
        &mut weights,
        "a.k_proj.weight",
        &[kv_size, hidden],
        &mut seed,
    );
    put(
        &mut weights,
        "a.v_proj.weight",
        &[kv_size, hidden],
        &mut seed,
    );
    put(
        &mut weights,
        "a.o_proj.weight",
        &[hidden, q_size],
        &mut seed,
    );
    put(
        &mut weights,
        "a.gate_proj.weight",
        &[q_size, hidden],
        &mut seed,
    );
    put(&mut weights, "a.q_norm.weight", &[head_dim], &mut seed);
    put(&mut weights, "a.k_norm.weight", &[head_dim], &mut seed);

    let x = mlxcel_core::from_slice_f32(&noise(hidden as usize, 0x1234_ABCD), &[1, 1, hidden]);
    let kv_shape = [1, args.num_key_value_heads as i32, 3, head_dim];
    let kv_len: usize = kv_shape.iter().product::<i32>() as usize;
    let keys = mlxcel_core::from_slice_f32(&noise(kv_len, 0x2222_3333), &kv_shape);
    let values = mlxcel_core::from_slice_f32(&noise(kv_len, 0x4444_5555), &kv_shape);

    let run = |uses_rope: bool, offset: i32| -> Vec<f32> {
        let attn = Attention::from_weights(&weights, &args, "a", uses_rope).expect("attention");
        let mut cache = StubCache {
            offset,
            keys: mlxcel_core::copy(&keys),
            values: mlxcel_core::copy(&values),
        };
        read_all(&attn.forward(&x, &mut cache as &mut dyn CacheInterface, None))
    };

    let nope_near = run(false, 0);
    let nope_far = run(false, 1000);
    for (i, (a, b)) in nope_near.iter().zip(nope_far.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "channel {i}: a full-attention layer moved from {a} to {b} when the sequence \
             offset changed, so it is applying RoPE. Upstream builds no rope on a global layer."
        );
    }

    let rope_near = run(true, 0);
    let rope_far = run(true, 1000);
    let moved = rope_near
        .iter()
        .zip(rope_far.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        moved,
        "a sliding layer must rotate, so its output has to depend on the offset; if this fails \
         the NoPE assertion above proves nothing"
    );
}

#[test]
fn the_mup_scale_actually_reaches_the_hidden_state() {
    // The scale is a config flag, not a tensor, so nothing in the checkpoint
    // disagrees if it is dropped. Run the same weights with it on and off.
    let args = small_args();
    let weights = filled_weights(&args);
    let with_mup = AfmoeModel::from_weights(&weights, &args).expect("builds");

    let mut off = small_args();
    off.mup_enabled = false;
    let without_mup = AfmoeModel::from_weights(&weights, &off).expect("builds");

    let tokens = mlxcel_core::from_slice_i32(&[3, 5], &[1, 2]);
    let mut caches = LanguageModel::make_caches(&with_mup);
    let a = read_all(&LanguageModel::forward(
        &with_mup,
        &tokens,
        &mut caches,
        None,
    ));
    let mut caches = LanguageModel::make_caches(&without_mup);
    let b = read_all(&LanguageModel::forward(
        &without_mup,
        &tokens,
        &mut caches,
        None,
    ));

    let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
    assert!(
        differs,
        "mup_enabled changed nothing, so the sqrt(hidden_size) scale is not reaching the stack"
    );
}
