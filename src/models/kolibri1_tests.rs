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

//! Reference-free tests for Kolibri-1: config refusals, the router contract,
//! NoPE full layers, the checkpoint sanitize, and the model-owned serving
//! contracts over a tiny random-weight model.

use super::{Kolibri1Model, ModelArgs};
use crate::models::ModelType;
use crate::models::afmoe::Attention;
use crate::models::detection::get_model_type;
use crate::models::gemma3::CacheInterface;
use crate::models::kolibri1_moe::route_logits;
use crate::models::kolibri1_weights::{sanitize_weights, validate_weights};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const LAYERS: usize = 5;

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "kolibri1",
        "architectures": ["Kolibri1ForCausalLM"],
        "hidden_size": 32,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 48,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": false,
        "sliding_window": 5,
        "layer_types": [
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention"
        ],
        "eos_token_id": 47,
        "head_dtype": "float32"
    })
}

fn args() -> ModelArgs {
    serde_json::from_value(config_json()).expect("the test config parses")
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

fn read_i32(array: &MlxArray) -> Vec<i32> {
    read_all(&mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32))
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

/// A checkpoint laid out as the released MLX conversion ships it: numbered
/// per-expert planes and the bias under `moe.router.expert_bias`. `experts`
/// planes are written per projection (fewer than `num_experts` makes a short
/// stack).
fn checkpoint(args: &ModelArgs, experts: usize) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let head_dim = args.head_dim as i32;
    let q = args.num_attention_heads as i32 * head_dim;
    let kv = args.num_key_value_heads as i32 * head_dim;
    let moe = args.moe_intermediate_size as i32;
    let shared = args.shared_expert_intermediate_size as i32;
    let mut shapes: Vec<(String, Vec<i32>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![args.vocab_size as i32, hidden],
        ),
        ("model.norm.weight".into(), vec![hidden]),
        (
            "lm_head.weight".into(),
            vec![args.vocab_size as i32, hidden],
        ),
    ];
    for l in 0..args.num_hidden_layers {
        let p = format!("model.layers.{l}");
        for (leaf, shape) in [
            ("self_attn.q_proj.weight", vec![q, hidden]),
            ("self_attn.k_proj.weight", vec![kv, hidden]),
            ("self_attn.v_proj.weight", vec![kv, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, q]),
            ("self_attn.q_norm.weight", vec![head_dim]),
            ("self_attn.k_norm.weight", vec![head_dim]),
            ("input_layernorm.weight", vec![hidden]),
            ("post_attn_norm.weight", vec![hidden]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("post_ffn_norm.weight", vec![hidden]),
            ("mlp.gate.weight", vec![args.num_experts as i32, hidden]),
            ("moe.router.expert_bias", vec![args.num_experts as i32]),
            ("mlp.shared_experts.gate_proj.weight", vec![shared, hidden]),
            ("mlp.shared_experts.up_proj.weight", vec![shared, hidden]),
            ("mlp.shared_experts.down_proj.weight", vec![hidden, shared]),
        ] {
            shapes.push((format!("{p}.{leaf}"), shape));
        }
        for e in 0..experts {
            shapes.push((
                format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                vec![moe, hidden],
            ));
            shapes.push((
                format!("{p}.mlp.experts.{e}.up_proj.weight"),
                vec![moe, hidden],
            ));
            shapes.push((
                format!("{p}.mlp.experts.{e}.down_proj.weight"),
                vec![hidden, moe],
            ));
        }
    }
    // Sorted so the seed sequence, and with it the model, is the same on
    // every run (a `HashMap` walk is not).
    shapes.sort();
    let mut weights = WeightMap::new();
    let mut seed = 0x5EED_1234u32;
    for (key, shape) in shapes {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n: i32 = shape.iter().product();
        let mut values = noise(n as usize, seed);
        if key.ends_with("norm.weight") {
            values.iter_mut().for_each(|v| *v += 1.0);
        }
        weights.insert(key, mlxcel_core::from_slice_f32(&values, &shape));
    }
    weights
}

fn model() -> Kolibri1Model {
    let args = args();
    let mut weights = checkpoint(&args, args.num_experts);
    Kolibri1Model::from_weights(&mut weights, &args).expect("the synthetic model builds")
}

// Config.

#[test]
fn the_released_config_shape_parses_and_validates() {
    let mut config = config_json();
    config["quantization"] = serde_json::json!({"mode": "affine", "bits": 4, "group_size": 64});
    let args: ModelArgs = serde_json::from_value(config).unwrap();
    args.validate().unwrap();
    assert_eq!(args.full_attention_index(), Some(2));
    assert_eq!(args.sliding_index(), Some(0));
    assert!(args.head_is_f32());
    assert_eq!(args.eos_token_ids(), vec![47]);
    assert_eq!((args.group_size(), args.bits()), (64, 4));
}

#[test]
fn rope_parameters_override_the_flat_rope_theta() {
    let mut config = config_json();
    config.as_object_mut().unwrap().remove("rope_theta");
    config["rope_parameters"] = serde_json::json!({"rope_type": "default", "rope_theta": 500000.0});
    let args: ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.rope_theta, 500_000.0);
}

#[test]
fn a_layer_types_list_of_the_wrong_length_is_refused() {
    for len in [LAYERS - 1, LAYERS + 1] {
        let err = parse(|c| {
            let kinds = vec![serde_json::json!("sliding_attention"); len];
            c["layer_types"] = serde_json::Value::Array(kinds);
        })
        .unwrap_err();
        assert!(err.contains("layer_types has"), "{err}");
    }
}

#[test]
fn an_unknown_layer_type_is_refused() {
    let err = parse(|c| c["layer_types"][3] = serde_json::json!("chunked_attention")).unwrap_err();
    assert!(err.contains("layer_types[3]"), "{err}");
}

#[test]
fn impossible_scalars_are_refused() {
    let err = parse(|c| c["num_experts_per_tok"] = serde_json::json!(5)).unwrap_err();
    assert!(err.contains("num_experts_per_tok"), "{err}");
    let err = parse(|c| c["sliding_window"] = serde_json::json!(0)).unwrap_err();
    assert!(
        err.contains("sliding_window") || err.contains("window"),
        "{err}"
    );
    let err = parse(|c| c["hidden_act"] = serde_json::json!("gelu")).unwrap_err();
    assert!(err.contains("hidden_act"), "{err}");
    let err = parse(|c| c["rope_scaling"] = serde_json::json!({"rope_type": "yarn"})).unwrap_err();
    assert!(err.contains("rope_scaling"), "{err}");
}

#[test]
fn kolibri1_is_detected_from_its_model_type() {
    let dir = std::env::temp_dir().join(format!("mlxcel-kolibri1-detect-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json().to_string()).unwrap();
    let detected = get_model_type(&dir);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(detected.unwrap(), ModelType::Kolibri1);
}

// Router.

fn route(logits: &[f32], bias: &[f32], top_k: i32, norm: bool) -> (Vec<i32>, Vec<f32>) {
    let n = logits.len() as i32;
    let logits = mlxcel_core::from_slice_f32(logits, &[1, n]);
    let bias = mlxcel_core::from_slice_f32(bias, &[n]);
    let (indices, weights) = route_logits(&logits, &bias, top_k, norm);
    (read_i32(&indices), read_all(&weights))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn the_router_selects_on_raw_logits_plus_bias_and_weights_with_the_unbiased_sigmoid() {
    // logits + bias = [4, 3, 2, 0] selects {0, 1}. sigmoid(logits) + bias =
    // [0.98, 0.95, 3.27, 0.5] would select {2, 0}: the DeepSeek-V3 rule picks a
    // different expert here, so a port using it fails this test.
    let logits = [4.0, 3.0, -1.0, 0.0];
    let bias = [0.0, 0.0, 3.0, 0.0];
    let (indices, weights) = route(&logits, &bias, 2, false);
    assert_eq!(indices, vec![0, 1], "selection must be on logits + bias");
    assert!((weights[0] - sigmoid(4.0)).abs() < 1e-6, "{weights:?}");
    assert!((weights[1] - sigmoid(3.0)).abs() < 1e-6, "{weights:?}");

    let mut wrong: Vec<(f32, i32)> = logits
        .iter()
        .zip(bias)
        .enumerate()
        .map(|(i, (l, b))| (sigmoid(*l) + b, i as i32))
        .collect();
    wrong.sort_by(|a, b| b.0.total_cmp(&a.0));
    let wrong: Vec<i32> = wrong.iter().take(2).map(|(_, i)| *i).collect();
    assert_ne!(
        indices, wrong,
        "the fixture must separate the two selection rules"
    );
}

#[test]
fn the_router_orders_by_selection_value_and_gathers_unbiased_weights() {
    // Selection = [1, 5.5, 2, 6]: descending order is 3, 1, 2. The weights are
    // the sigmoid of the raw logits at those indices, not of the biased values,
    // and they are not renormalized while `norm_topk_prob` is false.
    let logits = [1.0, 0.5, 2.0, -1.0];
    let bias = [0.0, 5.0, 0.0, 7.0];
    let (indices, weights) = route(&logits, &bias, 3, false);
    assert_eq!(indices, vec![3, 1, 2]);
    let expected = [sigmoid(-1.0), sigmoid(0.5), sigmoid(2.0)];
    for (w, e) in weights.iter().zip(expected) {
        assert!((w - e).abs() < 1e-6, "{weights:?} vs {expected:?}");
    }

    let (_, normed) = route(&logits, &bias, 3, true);
    let sum: f32 = expected.iter().sum();
    for (w, e) in normed.iter().zip(expected) {
        assert!((w - e / sum).abs() < 1e-6, "{normed:?}");
    }
}

// Attention.

/// Reports whatever offset the test asks for and returns fixed keys and
/// values, so a block runs at two positions over identical inputs.
struct StubCache {
    offset: i32,
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
}

impl CacheInterface for StubCache {
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

    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (keys, values) = self.update_and_fetch(k, v);
        mlxcel_core::layers::attention(q, &keys, &values, scale, mask, 0.0, 0)
    }
}

#[test]
fn full_attention_layers_apply_no_rope_and_sliding_layers_do() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = args();
    let weights = model_weights();
    let hidden = args.hidden_size as i32;
    let kv_shape = [1, args.num_key_value_heads as i32, 3, args.head_dim as i32];
    let kv_len = kv_shape.iter().product::<i32>() as usize;
    let x = mlxcel_core::from_slice_f32(&noise(hidden as usize, 0x1234_ABCD), &[1, 1, hidden]);
    let keys = mlxcel_core::from_slice_f32(&noise(kv_len, 0x2222_3333), &kv_shape);
    let values = mlxcel_core::from_slice_f32(&noise(kv_len, 0x4444_5555), &kv_shape);

    // The layer schedule decides the rotation: layer 2 is full, layer 0 slides.
    let model = model();
    assert!(
        !model.layers[2].self_attn.uses_rope,
        "a full layer must be NoPE"
    );
    assert!(
        model.layers[0].self_attn.uses_rope,
        "a sliding layer must rotate"
    );
    assert!(
        model.layers[2].self_attn.gate_proj.is_none(),
        "Kolibri-1 has no attention gate"
    );

    let run = |layer: usize, offset: i32| -> Vec<f32> {
        let uses_rope = args.is_sliding_layer(layer);
        let attn = Attention::load(
            &weights,
            &format!("model.layers.{layer}.self_attn"),
            &args.attention_geometry(),
            uses_rope,
            false,
        )
        .unwrap();
        let mut cache = StubCache {
            offset,
            keys: mlxcel_core::copy(&keys),
            values: mlxcel_core::copy(&values),
        };
        read_all(&attn.forward(&x, &mut cache as &mut dyn CacheInterface, None))
    };
    let (near, far) = (run(2, 0), run(2, 1000));
    for (i, (a, b)) in near.iter().zip(&far).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "channel {i}: the full layer moved from {a} to {b} with the offset, so it rotates"
        );
    }
    let (near, far) = (run(0, 0), run(0, 1000));
    assert!(
        near.iter().zip(&far).any(|(a, b)| (a - b).abs() > 1e-4),
        "a sliding layer's output must depend on the offset"
    );
}

/// The sanitized weights of [`model`], for building single blocks.
fn model_weights() -> WeightMap {
    let args = args();
    let mut weights = checkpoint(&args, args.num_experts);
    sanitize_weights(&mut weights, &args).unwrap();
    weights
}

// Sanitize.

#[test]
fn sanitize_renames_the_router_bias_and_stacks_the_expert_planes() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = args();
    let original = checkpoint(&args, args.num_experts);
    let mut weights = checkpoint(&args, args.num_experts);
    sanitize_weights(&mut weights, &args).unwrap();

    let p = "model.layers.3";
    assert!(!weights.contains_key(&format!("{p}.moe.router.expert_bias")));
    assert_eq!(
        read_all(&weights[&format!("{p}.mlp.gate.e_score_correction_bias")]),
        read_all(&original[&format!("{p}.moe.router.expert_bias")]),
    );
    assert!(
        !weights.keys().any(|k| k.contains(".mlp.experts.")),
        "planes must be drained"
    );
    for (proj, out, inp) in [
        ("gate_proj", 16, 32),
        ("up_proj", 16, 32),
        ("down_proj", 32, 16),
    ] {
        let stacked = &weights[&format!("{p}.mlp.switch_mlp.{proj}.weight")];
        assert_eq!(mlxcel_core::array_shape(stacked), vec![4, out, inp]);
        for e in 0..4 {
            let plane = mlxcel_core::slice(stacked, &[e, 0, 0], &[e + 1, out, inp]);
            assert_eq!(
                read_all(&plane),
                read_all(&original[&format!("{p}.mlp.experts.{e}.{proj}.weight")]),
                "plane {e} of {proj} must be expert {e}"
            );
        }
    }
    validate_weights(&weights, &args).unwrap();

    // Idempotent, and an already-stacked `mlp.experts.{proj}` export is renamed.
    let before = weights.len();
    sanitize_weights(&mut weights, &args).unwrap();
    assert_eq!(weights.len(), before);
    let moved = weights
        .remove(&format!("{p}.mlp.switch_mlp.up_proj.weight"))
        .unwrap();
    weights.insert(format!("{p}.mlp.experts.up_proj.weight"), moved);
    sanitize_weights(&mut weights, &args).unwrap();
    assert!(weights.contains_key(&format!("{p}.mlp.switch_mlp.up_proj.weight")));
    validate_weights(&weights, &args).unwrap();
}

#[test]
fn the_alternative_mlp_expert_bias_name_is_accepted() {
    let args = args();
    let mut weights = checkpoint(&args, args.num_experts);
    let bias = weights
        .remove("model.layers.0.moe.router.expert_bias")
        .unwrap();
    weights.insert("model.layers.0.mlp.expert_bias".into(), bias);
    sanitize_weights(&mut weights, &args).unwrap();
    assert!(weights.contains_key("model.layers.0.mlp.gate.e_score_correction_bias"));
}

#[test]
fn a_checkpoint_with_fewer_expert_planes_than_num_experts_is_refused() {
    let args = args();
    let mut weights = checkpoint(&args, args.num_experts - 1);
    let err = match Kolibri1Model::from_weights(&mut weights, &args) {
        Ok(_) => panic!("a short expert stack must be refused"),
        Err(err) => err,
    };
    assert!(err.contains("exceeds the 3 expert planes"), "{err}");
}

#[test]
fn an_expert_plane_of_another_shape_fails_the_load_instead_of_aborting() {
    // `mlxcel_core::stack` aborts the process on a shape mismatch, so the
    // sanitize pass must refuse the plane first (issue #3297).
    let args = args();
    let key = "model.layers.2.mlp.experts.1.down_proj.weight";
    let mut weights = checkpoint(&args, args.num_experts);
    let shape = mlxcel_core::array_shape(&weights[key]);
    weights.insert(
        key.into(),
        mlxcel_core::full_f32(&[shape[0], shape[1] + 1], 0.0, mlxcel_core::dtype::FLOAT32),
    );
    let err = sanitize_weights(&mut weights, &args).unwrap_err();
    assert!(err.contains(key), "{err}");

    let mut weights = checkpoint(&args, args.num_experts);
    weights.insert(
        key.into(),
        mlxcel_core::full_f32(&[shape[0], shape[1] + 1], 0.0, mlxcel_core::dtype::FLOAT32),
    );
    let err = match Kolibri1Model::from_weights(&mut weights, &args) {
        Ok(_) => panic!("a mismatched expert plane must be refused"),
        Err(err) => err,
    };
    assert!(err.contains(key), "{err}");
}

#[test]
fn a_quantized_router_is_refused() {
    let args = args();
    let mut weights = checkpoint(&args, args.num_experts);
    weights.insert(
        "model.layers.0.mlp.gate.scales".into(),
        mlxcel_core::full_f32(&[4, 1], 1.0, mlxcel_core::dtype::FLOAT32),
    );
    sanitize_weights(&mut weights, &args).unwrap();
    let err = validate_weights(&weights, &args).unwrap_err();
    assert!(err.contains("quantized"), "{err}");
}

// Model.

#[test]
fn the_synthetic_model_produces_finite_f32_logits_past_the_window() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let tokens: Vec<i32> = (0..12).map(|i| (i * 7 + 3) % 48).collect();
    let input = mlxcel_core::from_slice_i32(&tokens, &[1, 12]);
    let mut caches = model.make_caches();
    let logits = model.forward(&input, &mut caches, None);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 12, 48]);
    assert_eq!(
        mlxcel_core::array_dtype(&logits),
        mlxcel_core::dtype::FLOAT32
    );
    assert!(read_all(&logits).iter().all(|v| v.is_finite()));
}

#[test]
fn decoding_through_the_rotating_cache_matches_the_full_prefill() {
    // Twelve tokens through a window of five: the sliding rings wrap during
    // the token-by-token decode, which must still reproduce the one-shot
    // prefill's last-position logits.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let tokens: Vec<i32> = (0..12).map(|i| (i * 5 + 1) % 48).collect();
    let mut caches = model.make_caches();
    let full = model.forward(
        &mlxcel_core::from_slice_i32(&tokens, &[1, 12]),
        &mut caches,
        None,
    );
    let full_last = read_all(&mlxcel_core::slice(&full, &[0, 11, 0], &[1, 12, 48]));

    let seq = mlxcel_core::cache::SequenceId::from_raw(7);
    model.prepare_sequence_state(seq);
    let prefix = mlxcel_core::from_slice_i32(&tokens[..2], &[1, 2]);
    model.forward_with_sequence_id(&prefix, Some(seq), &mut caches, None);
    let mut last = Vec::new();
    for &token in &tokens[2..] {
        let step = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        last = read_all(&model.forward_with_sequence_id(&step, Some(seq), &mut caches, None));
    }
    for (i, (a, b)) in full_last.iter().zip(&last).enumerate() {
        assert!((a - b).abs() < 1e-3, "logit {i}: prefill {a} vs decode {b}");
    }
}

#[test]
fn kolibri1_serving_contracts() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let policy: &dyn LanguageModel = &model;
    assert_eq!(policy.eos_token_ids(), vec![47]);
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

    let foreign = mlxcel_core::generate::ModelStateSnapshot::new("afmoe", 4);
    assert!(model.restore_sequence_state(warm, &foreign).is_err());
    assert!(!model.snapshot_truncatable_to(&foreign, 2));
}

#[test]
fn model_owned_batched_multistep() {
    crate::models::batched_decode_harness::assert_model_owned_multistep(model);
}

#[test]
fn kolibri1_last_row_logits_match_the_full_forward() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::last_row_logits_moe_tests::assert_last_row_logits_under_sequence_ids(
        model,
        crate::models::last_row_logits_harness::TOKENS,
    );
}
