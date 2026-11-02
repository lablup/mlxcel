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

//! Checkpoint-free tests for the MiMo v2 Flash loader.
//!
//! The groups that a passing real-checkpoint run cannot distinguish from a
//! wrong implementation are covered first: the partial-RoPE width, the
//! per-layer RoPE base, the list-driven sliding/full schedule, the Q/K vs V
//! head-width split, the sliding-only sinks, and the `noaux_tc` router's
//! selection-vs-weighting contract.

use super::{MiMoV2FlashModel, ModelArgs, MoeGate, sanitize_weights};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// `mlx-community/MiMo-V2-Flash-4bit`'s published `config.json` (2026-11-02).
const PUBLISHED_CONFIG: &str = r#"{
    "add_full_attention_sink_bias": false,
    "add_swa_attention_sink_bias": true,
    "architectures": ["MiMoV2FlashForCausalLM"],
    "attention_bias": false,
    "attention_chunk_size": 128,
    "attention_dropout": 0.0,
    "attention_value_scale": 0.707,
    "head_dim": 192,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "hybrid_layer_pattern": [0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
    "initializer_range": 0.02,
    "intermediate_size": 16384,
    "layernorm_epsilon": 1e-05,
    "max_position_embeddings": 262144,
    "model_type": "mimo_v2_flash",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "n_group": 1,
    "n_routed_experts": 256,
    "n_shared_experts": null,
    "norm_topk_prob": true,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "partial_rotary_factor": 0.334,
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"},
    "rope_theta": 5000000,
    "routed_scaling_factor": null,
    "scoring_func": "sigmoid",
    "sliding_window": 128,
    "sliding_window_size": 128,
    "swa_head_dim": 192,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 8,
    "swa_rope_theta": 10000,
    "swa_v_head_dim": 128,
    "tie_word_embeddings": false,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "vocab_size": 152576,
    "v_head_dim": 128
}"#;

fn published_args() -> ModelArgs {
    serde_json::from_str(PUBLISHED_CONFIG).expect("the published config parses")
}

/// A shrunk MiMo v2 Flash: 4 layers alternating full/sliding/sliding/full,
/// 4 query heads of 8, a separate V width of 4, and 4 routed experts.
fn tiny_args() -> ModelArgs {
    ModelArgs {
        hidden_size: 32,
        num_hidden_layers: 4,
        intermediate_size: 64,
        moe_intermediate_size: 16,
        vocab_size: 48,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        v_head_dim: 4,
        swa_num_attention_heads: 4,
        swa_num_key_value_heads: 2,
        swa_head_dim: 8,
        swa_v_head_dim: 4,
        hybrid_layer_pattern: vec![0, 1, 1, 0],
        moe_layer_freq: vec![0, 1, 1, 1],
        sliding_window_size: 2,
        max_position_embeddings: 64,
        layernorm_epsilon: 1e-5,
        rope_theta: 10_000.0,
        swa_rope_theta: 100.0,
        partial_rotary_factor: 0.5,
        add_swa_attention_sink_bias: true,
        add_full_attention_sink_bias: false,
        attention_value_scale: Some(0.707),
        n_routed_experts: Some(4),
        num_experts_per_tok: 2,
        n_shared_experts: None,
        routed_scaling_factor: None,
        topk_method: "noaux_tc".to_string(),
        scoring_func: "sigmoid".to_string(),
        norm_topk_prob: true,
        n_group: 1,
        topk_group: 1,
        tie_word_embeddings: false,
        quantization: None,
        ..ModelArgs::default()
    }
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn lazy(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(shape, 0.0, mlxcel_core::dtype::FLOAT32)
}

fn synthetic_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
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
        let sliding = args.is_sliding_layer(layer);
        let (heads, kv_heads, head_dim, v_head_dim) = if sliding {
            (
                args.swa_num_attention_heads,
                args.swa_num_key_value_heads,
                args.swa_head_dim,
                args.swa_v_head_dim,
            )
        } else {
            (
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.v_head_dim,
            )
        };
        let q_width = (heads * head_dim) as i32;
        let kv_width = (kv_heads * head_dim) as i32;
        let v_width = (kv_heads * v_head_dim) as i32;
        let o_width = (heads * v_head_dim) as i32;
        weights.insert(format!("{attn}.q_proj.weight"), lazy(&[q_width, hidden]));
        weights.insert(format!("{attn}.k_proj.weight"), lazy(&[kv_width, hidden]));
        weights.insert(format!("{attn}.v_proj.weight"), lazy(&[v_width, hidden]));
        weights.insert(format!("{attn}.o_proj.weight"), lazy(&[hidden, o_width]));
        if sliding && args.add_swa_attention_sink_bias {
            weights.insert(format!("{attn}.attention_sink_bias"), lazy(&[heads as i32]));
        }
        weights.insert(format!("{prefix}.input_layernorm.weight"), lazy(&[hidden]));
        weights.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            lazy(&[hidden]),
        );

        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            let experts = args.num_routed_experts() as i32;
            let moe = args.moe_intermediate_size as i32;
            weights.insert(format!("{mlp}.gate.weight"), lazy(&[experts, hidden]));
            weights.insert(
                format!("{mlp}.gate.e_score_correction_bias"),
                lazy(&[experts]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.gate_proj.weight"),
                lazy(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.up_proj.weight"),
                lazy(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.down_proj.weight"),
                lazy(&[experts, hidden, moe]),
            );
        } else {
            let intermediate = args.intermediate_size as i32;
            weights.insert(
                format!("{mlp}.gate_proj.weight"),
                lazy(&[intermediate, hidden]),
            );
            weights.insert(
                format!("{mlp}.up_proj.weight"),
                lazy(&[intermediate, hidden]),
            );
            weights.insert(
                format!("{mlp}.down_proj.weight"),
                lazy(&[hidden, intermediate]),
            );
        }
    }
    weights
}

/// Replace every synthetic tensor with seeded noise so a forward pass exercises
/// real arithmetic rather than zeros.
fn filled_weights(args: &ModelArgs) -> WeightMap {
    let mut weights = synthetic_weights(args);
    let mut keys: Vec<String> = weights.keys().cloned().collect();
    // Sorted: `WeightMap` is a `HashMap`, and its iteration order is randomized
    // per process. An unsorted walk would build a different model each run.
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

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

#[test]
fn detection_and_registration_route_the_checkpoint_to_this_family() {
    use crate::models::ModelType;

    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), PUBLISHED_CONFIG).expect("write config");
    assert_eq!(
        crate::models::get_model_type(dir.path()).expect("detects"),
        ModelType::MiMoV2Flash
    );
    assert!(crate::model_metadata::is_config_backed_model_type(
        ModelType::MiMoV2Flash
    ));
    assert!(crate::model_metadata::has_config_backed_registration(
        ModelType::MiMoV2Flash
    ));
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[test]
fn published_config_parses_with_the_checkpoint_scalars() {
    let args = published_args();
    args.validate().expect("the published config is valid");

    assert_eq!(args.model_type, "mimo_v2_flash");
    assert_eq!(args.hidden_size, 4096);
    assert_eq!(args.num_hidden_layers, 48);
    assert_eq!(args.num_attention_heads, 64);
    assert_eq!(args.num_key_value_heads, 4);
    assert_eq!(args.head_dim, 192);
    assert_eq!(args.v_head_dim, 128);
    assert_eq!(args.swa_num_key_value_heads, 8);
    assert_eq!(args.swa_head_dim, 192);
    assert_eq!(args.swa_v_head_dim, 128);
    assert_eq!(args.partial_rotary_factor, 0.334);
    assert_eq!(args.attention_value_scale, Some(0.707));
    assert_eq!(args.n_routed_experts, Some(256));
    assert_eq!(args.n_shared_experts, None);
    assert_eq!(args.routed_scaling_factor, None);
    assert_eq!(args.routed_scaling_factor(), 1.0);
    assert_eq!(args.sliding_window_size, 128);
    assert_eq!(args.rope_theta, 5_000_000.0);
    assert_eq!(args.swa_rope_theta, 10_000.0);
    assert!(!args.tie_word_embeddings);
}

#[test]
fn a_config_with_missing_fields_still_deserializes_but_validation_explains_why() {
    let args: ModelArgs = serde_json::from_str("{}").expect("defaults make every field optional");
    assert_eq!(args.hidden_size, 4096);
    let err = args
        .validate()
        .expect_err("an empty config cannot describe the layer schedule");
    assert!(err.contains("hybrid_layer_pattern"), "{err}");
}

#[test]
fn partial_rotary_dims_are_floored_from_the_float_factor() {
    let args = published_args();
    // int(0.334 * 192) = 64. A `usize` field would fail to deserialize the
    // published float, and rounding instead of flooring also changes the width.
    assert_eq!(args.rope_dims_for(192).unwrap(), 64);
    assert_eq!(args.rope_dims_for(128).unwrap(), 42);
    // The two published head widths agree.
    assert_eq!(
        args.rope_dims_for(args.head_dim).unwrap(),
        args.rope_dims_for(args.swa_head_dim).unwrap()
    );
}

#[test]
fn the_rope_base_is_selected_per_layer_from_the_pattern() {
    let args = published_args();
    // Full layers in the published list: 0, then every sixth layer from 5.
    for layer in [0, 5, 11, 17, 23, 29, 35, 41, 47] {
        assert!(
            !args.is_sliding_layer(layer),
            "layer {layer} should be full"
        );
        assert_eq!(args.rope_base_for(layer), 5_000_000.0, "layer {layer}");
    }
    // Everything else slides.
    for layer in [1, 2, 3, 4, 6, 10, 44, 46] {
        assert!(args.is_sliding_layer(layer), "layer {layer} should slide");
        assert_eq!(args.rope_base_for(layer), 10_000.0, "layer {layer}");
    }
}

#[test]
fn the_schedule_is_read_from_the_lists_not_from_a_modulus() {
    let args = published_args();
    let sliding = (0..args.num_hidden_layers)
        .filter(|&i| args.is_sliding_layer(i))
        .count();
    assert_eq!(sliding, 39);
    assert_eq!(args.num_hidden_layers - sliding, 9);

    // Layer 0 is the dense prefix; every other layer is sparse.
    assert!(!args.is_moe_layer(0));
    for layer in 1..args.num_hidden_layers {
        assert!(args.is_moe_layer(layer), "layer {layer} should be MoE");
    }

    // Swap two entries and the schedule follows the list, not a period.
    let mut swapped = published_args();
    swapped.hybrid_layer_pattern.swap(1, 5);
    assert!(swapped.is_sliding_layer(5));
    assert!(!swapped.is_sliding_layer(1));
}

#[test]
fn validation_rejects_a_broken_schedule_and_broken_routing() {
    let mut short_pattern = published_args();
    short_pattern.hybrid_layer_pattern.pop();
    assert!(
        short_pattern
            .validate()
            .expect_err("a short pattern must be rejected")
            .contains("hybrid_layer_pattern")
    );

    let mut short_freq = published_args();
    short_freq.moe_layer_freq.pop();
    assert!(
        short_freq
            .validate()
            .expect_err("a short MoE schedule must be rejected")
            .contains("moe_layer_freq")
    );

    let mut bad_topk = published_args();
    bad_topk.num_experts_per_tok = 999;
    assert!(bad_topk.validate().is_err());

    let mut bad_scoring = published_args();
    bad_scoring.scoring_func = "softmax".to_string();
    assert!(bad_scoring.validate().is_err());

    let mut bad_topk_method = published_args();
    bad_topk_method.topk_method = "greedy".to_string();
    assert!(bad_topk_method.validate().is_err());
}

#[test]
fn missing_experts_with_a_sparse_schedule_is_a_load_error() {
    let mut args = published_args();
    args.n_routed_experts = None;
    let err = args
        .validate()
        .expect_err("a sparse schedule without routed experts would load all-dense");
    assert!(err.contains("n_routed_experts"), "{err}");
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build a gate whose bias and parameters are explicit, so `route_scores` can
/// be driven with hand-picked scores.
fn gate_with_bias(args: &ModelArgs, bias: &[f32], norm: bool, scaling: f32) -> MoeGate {
    let mut weights = WeightMap::new();
    let experts = args.num_routed_experts() as i32;
    weights.insert(
        "g.gate.weight".into(),
        mlxcel_core::from_slice_f32(
            &noise((experts * args.hidden_size as i32) as usize, 7),
            &[experts, args.hidden_size as i32],
        ),
    );
    weights.insert(
        "g.gate.e_score_correction_bias".into(),
        mlxcel_core::from_slice_f32(bias, &[experts]),
    );
    let mut gate = MoeGate::from_weights(&weights, args, "g").expect("gate loads");
    gate.norm_topk_prob = norm;
    gate.routed_scaling_factor = scaling;
    gate
}

fn sorted_pairs(indices: &MlxArray, weights: &MlxArray) -> Vec<(i32, f32)> {
    let mut pairs: Vec<(i32, f32)> = read_all(indices)
        .into_iter()
        .map(|i| i as i32)
        .zip(read_all(weights))
        .collect();
    pairs.sort_by_key(|(index, _)| *index);
    pairs
}

#[test]
fn routing_selects_on_biased_scores_but_weights_from_unbiased_ones() {
    let args = tiny_args();
    // sigmoid-like scores; the bias promotes expert 2 above both 1 and 0.
    let scores = mlxcel_core::from_slice_f32(&[0.9, 0.8, 0.7, 0.6], &[1, 4]);
    let gate = gate_with_bias(&args, &[0.0, 0.0, 0.5, 0.0], true, 1.0);

    let (indices, weights) = gate.route_scores(&scores);
    let pairs = sorted_pairs(&indices, &weights);

    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, 0, "expert 0 must be selected");
    assert_eq!(pairs[1].0, 2, "the bias must pull expert 2 into the top-2");
    // The weights come from the UNBIASED 0.9 / 0.7, not from 0.9 / 1.2.
    assert!((pairs[0].1 - 0.5625).abs() < 1e-5, "{:?}", pairs);
    assert!((pairs[1].1 - 0.4375).abs() < 1e-5, "{:?}", pairs);
}

#[test]
fn routing_normalization_and_scaling_are_independent_switches() {
    let args = tiny_args();
    let scores = mlxcel_core::from_slice_f32(&[0.9, 0.8, 0.7, 0.6], &[1, 4]);

    // norm off, scale 2: raw unbiased scores times the scaling factor.
    let gate = gate_with_bias(&args, &[0.0, 0.0, 0.5, 0.0], false, 2.0);
    let (indices, weights) = gate.route_scores(&scores);
    let pairs = sorted_pairs(&indices, &weights);
    assert_eq!(pairs[0].0, 0);
    assert_eq!(pairs[1].0, 2);
    assert!((pairs[0].1 - 1.8).abs() < 1e-5, "{:?}", pairs);
    assert!((pairs[1].1 - 1.4).abs() < 1e-5, "{:?}", pairs);
}

#[test]
fn the_correction_bias_is_loaded_in_float32() {
    let args = tiny_args();
    let mut weights = WeightMap::new();
    let experts = args.num_routed_experts() as i32;
    weights.insert(
        "g.gate.weight".into(),
        lazy(&[experts, args.hidden_size as i32]),
    );
    weights.insert("g.gate.e_score_correction_bias".into(), lazy(&[experts]));
    let gate = MoeGate::from_weights(&weights, &args, "g").expect("gate loads");
    assert_eq!(
        mlxcel_core::array_dtype(&gate.e_score_correction_bias),
        mlxcel_core::dtype::FLOAT32
    );
}

#[test]
fn a_missing_correction_bias_defaults_to_zeros() {
    let args = tiny_args();
    let mut weights = WeightMap::new();
    let experts = args.num_routed_experts() as i32;
    weights.insert(
        "g.gate.weight".into(),
        lazy(&[experts, args.hidden_size as i32]),
    );
    let gate = MoeGate::from_weights(&weights, &args, "g").expect("gate loads");
    assert!(
        read_all(&gate.e_score_correction_bias)
            .iter()
            .all(|v| *v == 0.0)
    );
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[test]
fn a_synthetic_model_builds_and_produces_finite_logits() {
    let args = tiny_args();
    let model = MiMoV2FlashModel::from_weights(&filled_weights(&args), &args)
        .expect("the tiny model builds");
    assert_eq!(model.num_layers(), args.num_hidden_layers);

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let mut caches = LanguageModel::make_caches(&model);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, args.vocab_size as i32]
    );
    assert!(read_all(&logits).iter().all(|v| v.is_finite()));
}

#[test]
fn sliding_layers_carry_sinks_and_full_layers_do_not() {
    let args = tiny_args();
    let model = MiMoV2FlashModel::from_weights(&filled_weights(&args), &args)
        .expect("the tiny model builds");
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        assert_eq!(
            layer.self_attn.sinks.is_some(),
            args.is_sliding_layer(layer_idx),
            "layer {layer_idx} sink presence must follow the schedule"
        );
    }
}

#[test]
fn the_qk_and_v_head_widths_stay_separate() {
    let args = tiny_args();
    let model = MiMoV2FlashModel::from_weights(&filled_weights(&args), &args)
        .expect("the tiny model builds");
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let attn = &layer.self_attn;
        assert_eq!(attn.head_dim, 8, "layer {layer_idx}");
        assert_eq!(attn.v_head_dim, 4, "layer {layer_idx}");
        assert_ne!(attn.head_dim, attn.v_head_dim, "layer {layer_idx}");
    }
}

#[test]
fn the_prefill_is_causal_even_though_generation_passes_no_mask() {
    let args = tiny_args();
    let weights = filled_weights(&args);
    let model = MiMoV2FlashModel::from_weights(&weights, &args).expect("builds");
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
            "logit {i} at position 0 changed when later tokens were added: {a} vs {b}. The \
             prefill is not causal."
        );
    }
}

#[test]
fn attention_value_scale_reaches_the_forward_pass() {
    let args = tiny_args();
    let weights = filled_weights(&args);

    let mut scaled_args = tiny_args();
    scaled_args.attention_value_scale = Some(0.707);
    let scaled = MiMoV2FlashModel::from_weights(&weights, &scaled_args).expect("builds");

    let mut unscaled_args = tiny_args();
    unscaled_args.attention_value_scale = None;
    let unscaled = MiMoV2FlashModel::from_weights(&weights, &unscaled_args).expect("builds");

    let tokens = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut caches = LanguageModel::make_caches(&scaled);
    let a = read_all(&LanguageModel::forward(&scaled, &tokens, &mut caches, None));
    let mut caches = LanguageModel::make_caches(&unscaled);
    let b = read_all(&LanguageModel::forward(
        &unscaled,
        &tokens,
        &mut caches,
        None,
    ));

    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
        "changing attention_value_scale must change the output; otherwise V is never scaled"
    );
}

#[test]
fn sanitize_is_a_pass_through_and_drops_a_future_mtp_block() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.switch_mlp.gate_proj.weight".into(),
        lazy(&[4, 16, 32]),
    );
    weights.insert(
        "model.layers.0.self_attn.attention_sink_bias".into(),
        lazy(&[4]),
    );
    weights.insert("model.mtp.foo.weight".into(), lazy(&[2, 2]));

    sanitize_weights(&mut weights);

    assert!(weights.contains_key("model.layers.0.mlp.switch_mlp.gate_proj.weight"));
    assert!(weights.contains_key("model.layers.0.self_attn.attention_sink_bias"));
    assert!(!weights.contains_key("model.mtp.foo.weight"));
}
