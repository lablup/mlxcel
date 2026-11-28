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

//! PLaMo 3 tests: the layer pattern, the six norm offsets, Q/K norm before
//! RoPE, the YaRN full-layer table against the reference formula, config
//! refusals, detection, and the model-owned serving contracts over a tiny
//! random-weight model.

use super::{Plamo3Model, is_full_attention, sliding_prefill_mask};
use crate::models::ModelType;
use crate::models::detection::get_model_type;
use crate::models::plamo3_config::ModelArgs;
use crate::models::plamo3_layers::norm_offsets;
use crate::models::rope_utils::RopeScalingKind;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const LAYERS: usize = 5;
const VOCAB: i32 = 48;

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "plamo3",
        "architectures": ["Plamo3ForCausalLM"],
        "hidden_size": 32,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 16,
        "vocab_size": VOCAB,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "window_size": 5,
        "sliding_window_pattern": 4,
        "rope_theta": 1000000,
        "rope_local_theta": 10000,
        "rope_scaling_factor": 256.0,
        "initial_context_length": 4096,
        "scale_embedding": false,
        "tie_word_embeddings": false,
        "image_token_id": null,
        "image_feature_size": null,
        "image_proj_type": "linear",
        "linear_type": "normal",
        "eos_token_id": [2, 16]
    })
}

fn args_from(config: serde_json::Value) -> ModelArgs {
    serde_json::from_value(config).expect("the test config parses")
}

fn parse(mutate: impl FnOnce(&mut serde_json::Value)) -> Result<(), String> {
    let mut config = config_json();
    mutate(&mut config);
    let args: ModelArgs = serde_json::from_value(config).map_err(|e| e.to_string())?;
    args.validate()
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

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::astype(
        &mlxcel_core::reshape(array, &[-1]),
        mlxcel_core::dtype::FLOAT32,
    );
    mlxcel_core::eval(&flat);
    let n = mlxcel_core::array_shape(&flat)[0];
    (0..n)
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(&flat, &[i], &[i + 1])))
        .collect()
}

/// A checkpoint in the released key layout (note `model.layers.layers.N`).
/// Norm weights are `zero_norms ? 0 : noise`, so a zero checkpoint exposes
/// the folded offsets directly.
fn checkpoint(args: &ModelArgs, zero_norms: bool) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let d = args.head_dim as i32;
    let q = args.num_attention_heads as i32 * d;
    let kv = args.num_key_value_heads as i32 * d;
    let inter = args.intermediate_size as i32;
    let vocab = args.vocab_size as i32;
    let mut shapes: Vec<(String, Vec<i32>)> = vec![
        ("model.embed_tokens.weight".into(), vec![vocab, hidden]),
        ("model.norm.weight".into(), vec![hidden]),
        ("lm_head.weight".into(), vec![vocab, hidden]),
    ];
    for l in 0..args.num_hidden_layers {
        let p = format!("model.layers.layers.{l}");
        for (leaf, shape) in [
            ("mixer.qkv_proj.weight", vec![q + 2 * kv, hidden]),
            ("mixer.o_proj.weight", vec![hidden, q]),
            ("mixer.q_norm.weight", vec![d]),
            ("mixer.k_norm.weight", vec![d]),
            ("pre_mixer_norm.weight", vec![hidden]),
            ("post_mixer_norm.weight", vec![hidden]),
            ("pre_mlp_norm.weight", vec![hidden]),
            ("post_mlp_norm.weight", vec![hidden]),
            ("mlp.gate_up_proj.weight", vec![2 * inter, hidden]),
            ("mlp.down_proj.weight", vec![hidden, inter]),
        ] {
            shapes.push((format!("{p}.{leaf}"), shape));
        }
    }
    shapes.sort();
    let mut weights = WeightMap::new();
    let mut seed = 0x9A3D_0003u32;
    for (key, shape) in shapes {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n: i32 = shape.iter().product();
        let mut values = noise(n as usize, seed);
        if key.ends_with("norm.weight") && zero_norms {
            values.iter_mut().for_each(|v| *v = 0.0);
        }
        weights.insert(key, mlxcel_core::from_slice_f32(&values, &shape));
    }
    weights
}

fn model_from(config: serde_json::Value) -> Plamo3Model {
    let args = args_from(config);
    Plamo3Model::from_weights(&checkpoint(&args, false), &args).expect("the synthetic model builds")
}

fn model() -> Plamo3Model {
    model_from(config_json())
}

// Config.

#[test]
fn pattern_eight_makes_every_eighth_layer_full() {
    let full: Vec<usize> = (0..16).filter(|&i| is_full_attention(8, i)).collect();
    assert_eq!(full, vec![7, 15]);
    let mut config = config_json();
    config["num_hidden_layers"] = serde_json::json!(16);
    config["sliding_window_pattern"] = serde_json::json!(8);
    let args = args_from(config);
    assert_eq!(args.full_attention_index(), Some(7));
    assert_eq!(args.sliding_index(), Some(0));
    assert!(!args.is_full_layer(8));
}

#[test]
fn the_legacy_sliding_window_and_rope_global_theta_spellings_are_honored() {
    let mut config = config_json();
    config["sliding_window"] = serde_json::json!(64);
    assert_eq!(args_from(config.clone()).sliding_window(), 64);
    config["sliding_window"] = serde_json::json!([32, 32, null, 32]);
    assert_eq!(args_from(config.clone()).sliding_window(), 32);
    config.as_object_mut().unwrap().remove("sliding_window");
    config.as_object_mut().unwrap().remove("rope_theta");
    config["rope_global_theta"] = serde_json::json!(500000.0);
    assert_eq!(args_from(config).full_rope_theta(), 500000.0);
}

#[test]
fn unsupported_configs_are_refused_by_name() {
    let err = parse(|c| c["scale_embedding"] = serde_json::json!(true)).unwrap_err();
    assert!(err.contains("scale_embedding"), "{err}");
    let err = parse(|c| c["image_token_id"] = serde_json::json!(100)).unwrap_err();
    assert!(err.contains("image_token_id"), "{err}");
    let err = parse(|c| c["image_feature_size"] = serde_json::json!(1152)).unwrap_err();
    assert!(err.contains("image_feature_size"), "{err}");
    let err = parse(|c| c["linear_type"] = serde_json::json!("fp8")).unwrap_err();
    assert!(err.contains("linear_type"), "{err}");
    let err = parse(|c| c["initial_context_length"] = serde_json::Value::Null).unwrap_err();
    assert!(err.contains("initial_context_length"), "{err}");
    let err = parse(|c| c["window_size"] = serde_json::json!(0)).unwrap_err();
    assert!(err.contains("PLaMo 3"), "{err}");
    let err = parse(|c| c["num_key_value_heads"] = serde_json::json!(3)).unwrap_err();
    assert!(err.contains("num_key_value_heads"), "{err}");
    parse(|c| c["rope_scaling_factor"] = serde_json::json!(1.0)).unwrap();
}

#[test]
fn plamo3_is_detected_from_its_model_type() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), config_json().to_string()).unwrap();
    assert_eq!(get_model_type(dir.path()).unwrap(), ModelType::Plamo3);
}

// Norm offsets.

#[test]
fn every_norm_offset_is_folded_into_its_weight() {
    // Zero learned weights leave exactly the offset in each folded norm, so
    // dropping (or swapping) any of the six offsets changes a value here.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = args_from(config_json());
    let model = Plamo3Model::from_weights(&checkpoint(&args, true), &args).unwrap();
    let offsets: std::collections::HashMap<_, _> = norm_offsets().into_iter().collect();
    assert_eq!(offsets["pre_mixer_norm"], 1.0);
    assert!((offsets["post_mixer_norm"] - 0.2).abs() < 1e-7);
    assert_eq!(offsets["pre_mlp_norm"], 1.0);
    assert!((offsets["post_mlp_norm"] - 0.089_442_72).abs() < 1e-6);
    assert_eq!(offsets["qk_norm"], 1.0);
    assert_eq!(offsets["norm"], 1.0);

    let check = |name: &str, weight: &MlxArray, expected: f32| {
        for v in read_all(weight) {
            assert!((v - expected).abs() < 1e-6, "{name}: {v} vs {expected}");
        }
    };
    for layer in &model.layers {
        check("pre_mixer_norm", &layer.pre_mixer_norm.weight, 1.0);
        check("post_mixer_norm", &layer.post_mixer_norm.weight, 0.2);
        check("pre_mlp_norm", &layer.pre_mlp_norm.weight, 1.0);
        check("post_mlp_norm", &layer.post_mlp_norm.weight, 0.089_442_72);
        check("q_norm", &layer.mixer.q_norm.weight, 1.0);
        check("k_norm", &layer.mixer.k_norm.weight, 1.0);
    }
    check("norm", &model.norm.weight, 1.0);
}

// Attention.

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    read_all(a)
        .iter()
        .zip(read_all(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn qk_norms_run_before_rope_on_sliding_and_full_layers() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let x = mlxcel_core::from_slice_f32(&noise(3 * 32, 77), &[1, 3, 32]);
    // Layer 0 slides (plain RoPE), layer 3 is full (YaRN table and mscale).
    for layer_idx in [0, 3] {
        let attn = &model.layers[layer_idx].mixer;
        let offset = 6;
        let (q, k, _) = attn.rotated_qk(&x, offset);

        let qkv = attn.qkv_proj.forward(&x);
        let heads = |start: i32, n: i32| {
            let part = mlxcel_core::utils::slice_axis(&qkv, -1, start, start + n * 8);
            let part = mlxcel_core::reshape(&part, &[1, 3, n, 8]);
            mlxcel_core::transpose_axes(&part, &[0, 2, 1, 3])
        };
        let raw_q = heads(0, 4);
        let raw_k = heads(32, 1);
        let norm_then_rope = attn.rope.rotate(&attn.q_norm.forward(&raw_q), 8, offset);
        let rope_then_norm = attn.q_norm.forward(&attn.rope.rotate(&raw_q, 8, offset));
        assert!(
            max_abs_diff(&q, &norm_then_rope) < 1e-5,
            "layer {layer_idx}"
        );
        assert!(
            max_abs_diff(&norm_then_rope, &rope_then_norm) > 1e-2,
            "the fixture must separate the two orders on layer {layer_idx}"
        );
        let k_expected = attn.rope.rotate(&attn.k_norm.forward(&raw_k), 8, offset);
        assert!(max_abs_diff(&k, &k_expected) < 1e-5, "layer {layer_idx}");
    }
}

// YaRN.

/// `RotaryEmbedding._compute_buffers` from the checkpoint's
/// `modeling_plamo.py`, in f64: inverse frequencies and concentration.
fn reference_yarn(dim: usize, base: f64, factor: f64, original: f64) -> (Vec<f64>, f64) {
    let d_h = dim / 2;
    let low = d_h as f64 * (original / (32.0 * 2.0 * std::f64::consts::PI)).ln() / base.ln();
    let high = d_h as f64 * (original / (1.0 * 2.0 * std::f64::consts::PI)).ln() / base.ln();
    let inv = (0..d_h)
        .map(|i| {
            let freq = base.powf((2 * i) as f64 / dim as f64);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            (1.0 / (factor * freq)) * (1.0 - mask) + (1.0 / freq) * mask
        })
        .collect();
    (inv, 0.1 * factor.ln() + 1.0)
}

#[test]
fn the_full_layer_yarn_table_matches_the_reference_formula() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // The released 610m geometry: head_dim 128, rope_theta 1e6, factor 256
    // over 4096.
    let mut config = config_json();
    config["head_dim"] = serde_json::json!(128);
    let args = args_from(config);
    let spec = args
        .full_attention_rope_spec()
        .expect("factor 256 declares YaRN");
    assert_eq!(spec.truncate, Some(false));
    let kind = Plamo3Model::full_attention_rope(&args).unwrap();
    let RopeScalingKind::Yarn { freqs, mscale } = kind else {
        panic!("the full layers must build a YaRN table");
    };
    let (inv, concentration) = reference_yarn(128, 1e6, 256.0, 4096.0);
    assert!((f64::from(mscale) - concentration).abs() < 1e-5, "{mscale}");
    for (i, (divisor, want)) in read_all(&freqs).iter().zip(&inv).enumerate() {
        let got = 1.0 / f64::from(*divisor);
        assert!(
            (got - want).abs() <= 1e-5 * want.abs(),
            "pair {i}: {got} vs {want}"
        );
    }

    let mut plain = config_json();
    plain["rope_scaling_factor"] = serde_json::json!(1.0);
    plain["initial_context_length"] = serde_json::Value::Null;
    let args = args_from(plain);
    assert!(args.full_attention_rope_spec().is_none());
    assert!(matches!(
        Plamo3Model::full_attention_rope(&args).unwrap(),
        RopeScalingKind::Default
    ));
}

// Masks.

#[test]
fn the_sliding_mask_spans_window_plus_one_over_the_rotating_keys() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    assert!(sliding_prefill_mask(1, 0, 4).is_none());
    // Fresh prefill, window 2: the last row attends keys 1, 2, 3 (three keys).
    let mask = sliding_prefill_mask(4, 0, 2).unwrap();
    assert_eq!(mlxcel_core::array_shape(&mask), vec![4, 4]);
    let last_row: Vec<bool> = read_all(&mask)[12..]
        .iter()
        .map(|v| v.is_finite())
        .collect();
    assert_eq!(last_row, vec![false, true, true, true]);
    // Continuation after 10 tokens on a ring of 5: four retained prior keys.
    let mask = sliding_prefill_mask(3, 10, 5).unwrap();
    assert_eq!(mlxcel_core::array_shape(&mask), vec![3, 7]);
}

// Model.

fn forward_tokens(model: &Plamo3Model, tokens: &[i32]) -> UniquePtr<MlxArray> {
    let mut caches = model.make_caches();
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    model.forward(&input, &mut caches, None)
}

#[test]
fn a_prompt_past_the_window_wraps_the_ring_and_keeps_decoding() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let tokens: Vec<i32> = (0..12).map(|i| (i * 7 + 3) % VOCAB).collect();
    let logits = forward_tokens(&model, &tokens);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 12, VOCAB]);
    assert!(read_all(&logits).iter().all(|v| v.is_finite()));

    // Prefill 7, append a 3-token block (the continuation mask), then decode
    // single tokens through the wrapped ring.
    let mut caches = model.make_caches();
    let seq = mlxcel_core::cache::SequenceId::from_raw(11);
    model.prepare_sequence_state(seq);
    let feed = |slice: &[i32], caches: &mut Vec<_>| {
        let input = mlxcel_core::from_slice_i32(slice, &[1, slice.len() as i32]);
        model.forward_with_sequence_id(&input, Some(seq), caches, None)
    };
    feed(&tokens[..7], &mut caches);
    let block = feed(&tokens[7..10], &mut caches);
    assert_eq!(mlxcel_core::array_shape(&block), vec![1, 3, VOCAB]);
    for &t in &tokens[10..] {
        let step = feed(&[t], &mut caches);
        assert!(read_all(&step).iter().all(|v| v.is_finite()));
    }
}

#[test]
fn decode_matches_prefill_within_the_window() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let tokens = [5, 9, 1, 30, 17];
    let full = forward_tokens(&model, &tokens);
    let full_last = read_all(&mlxcel_core::slice(&full, &[0, 4, 0], &[1, 5, VOCAB]));

    let mut caches = model.make_caches();
    let seq = mlxcel_core::cache::SequenceId::from_raw(12);
    model.prepare_sequence_state(seq);
    let mut last = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        let input = mlxcel_core::from_slice_i32(&[t], &[1, 1]);
        let out = model.forward_with_sequence_id(&input, Some(seq), &mut caches, None);
        if i == tokens.len() - 1 {
            last = read_all(&out);
        }
    }
    for (i, (a, b)) in full_last.iter().zip(&last).enumerate() {
        assert!((a - b).abs() < 1e-3, "logit {i}: prefill {a} vs decode {b}");
    }
}

#[test]
fn all_full_and_all_sliding_stacks_run() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // Pattern 1: every layer is full and there is no sliding mask.
    let mut config = config_json();
    config["sliding_window_pattern"] = serde_json::json!(1);
    let model = model_from(config);
    assert!(model.layers.iter().all(|l| l.is_full));
    assert!(
        read_all(&forward_tokens(&model, &[1, 2, 3, 4, 5, 6, 7]))
            .iter()
            .all(|v| v.is_finite())
    );
    // Fewer layers than the pattern: every layer slides and there is no
    // full-attention layer.
    let mut config = config_json();
    config["sliding_window_pattern"] = serde_json::json!(8);
    let model = model_from(config);
    assert!(model.layers.iter().all(|l| !l.is_full));
    assert!(
        read_all(&forward_tokens(&model, &[1, 2, 3, 4, 5, 6, 7]))
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
fn plamo3_serving_contracts() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let policy: &dyn LanguageModel = &model;
    assert_eq!(policy.eos_token_ids(), vec![2, 16]);
    assert!(!policy.supports_chunked_prefill());
    assert!(!policy.supports_padded_prefill());
    assert!(policy.supports_snapshot_reuse());
    assert_eq!(policy.num_layers(), LAYERS);
}

#[test]
fn snapshot_restore_matches_cold_decode_and_refuses_a_foreign_family() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    use mlxcel_core::cache::SequenceId;
    let model = model();
    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(&[3, 9, 27, 4], &[1, 4]);
    let step = mlxcel_core::from_slice_i32(&[11], &[1, 1]);

    let cold = SequenceId::from_raw(1);
    model.prepare_sequence_state(cold);
    model.forward_with_sequence_id(&prompt, Some(cold), &mut caches, None);
    let snapshot = model.snapshot_sequence_state(cold, 4).expect("a snapshot");
    let expected = read_all(&model.forward_with_sequence_id(&step, Some(cold), &mut caches, None));

    let warm = SequenceId::from_raw(2);
    model.restore_sequence_state(warm, &snapshot).unwrap();
    let got = read_all(&model.forward_with_sequence_id(&step, Some(warm), &mut caches, None));
    for (a, b) in expected.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    let foreign = mlxcel_core::generate::ModelStateSnapshot::new("kolibri1", 4);
    assert!(model.restore_sequence_state(warm, &foreign).is_err());
}

#[test]
fn model_owned_batched_multistep() {
    crate::models::batched_decode_harness::assert_model_owned_multistep(model);
}

#[test]
fn plamo3_last_row_logits_match_the_full_forward() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::last_row_logits_moe_tests::assert_last_row_logits_under_sequence_ids(
        model,
        crate::models::last_row_logits_harness::TOKENS,
    );
}
