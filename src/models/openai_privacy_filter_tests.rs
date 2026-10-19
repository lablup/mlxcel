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

//! Config parsing, YaRN against upstream, the banded attention against a
//! full mask, chunk-invariance of attention and routing on a tiny random
//! model, the fused-weight sanitizer, and the generation / embedding
//! dispatch refusing the checkpoint.

use mlxcel_core::dtype;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use super::*;
use crate::models::embedding_test_support::mlx_test_guard;
use crate::models::openai_privacy_filter_layers::{PrivacyFilterMoe, band_mask, banded_attention};

/// `mlx-community/openai-privacy-filter-bf16`'s `config.json`, verbatim.
fn checkpoint_config() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/openai_privacy_filter_config.json"
    ))
    .expect("fixture parses")
}

fn args_from(config: &Value) -> PrivacyFilterArgs {
    PrivacyFilterArgs::from_json(&config.to_string()).expect("config parses")
}

#[test]
fn checkpoint_config_parses_with_the_lifted_rope_block() {
    let args = args_from(&checkpoint_config());
    assert_eq!(args.rope_theta, 150_000.0);
    assert_eq!(args.default_n_ctx, 128_000);
    assert_eq!(args.attention_chunk_size, 4096);
    assert_eq!(args.moe_chunk_size, 2048);
    assert_eq!(args.sliding_window, 128);
    assert_eq!(
        (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim
        ),
        (14, 2, 64)
    );
    let scaling = args.rope_scaling.as_ref().expect("yarn block lifted");
    assert_eq!(scaling["type"], json!("yarn"));
    assert_eq!(scaling["truncate"], json!(false));

    // `PrivacyFilterRoPE(config)` in mlx-vlm 0.7.8 for this config.
    let (freqs, factor) = args.yarn();
    assert_eq!(freqs.len(), 32);
    for (index, want) in [
        (0, 1.0f32),
        (9, 31.540_073),
        (17, 7_732.834),
        (18, 26_103.654),
        (31, 3_307_412.8),
    ] {
        assert!(
            (freqs[index] - want).abs() <= 2e-5 * want,
            "freq {index}: {} vs upstream {want}",
            freqs[index]
        );
    }
    assert!((factor - 1.346_573_6).abs() < 1e-6, "{factor}");

    // Unquantized: the quantization pair is only consulted when a plane
    // carries `.scales`, and then the shared loaders infer the mode.
    assert_eq!(
        args.quant_for("model.layers.0.mlp.router"),
        (64, 4, "affine".to_string())
    );
}

#[test]
fn eight_bit_config_quantizes_the_router_per_prefix() {
    let mut config = checkpoint_config();
    config["quantization"] = json!({
        "group_size": 64, "bits": 8, "mode": "affine",
        "model.layers.0.mlp.router": {"group_size": 64, "bits": 8},
        "model.layers.1.mlp.router": {"group_size": 32, "bits": 4},
    });
    let args = args_from(&config);
    assert_eq!(args.expert_quantization(), (64, 8, "affine".to_string()));
    assert_eq!(
        args.quant_for("model.layers.1.mlp.router"),
        (32, 4, "affine".to_string())
    );
    assert_eq!(
        args.quant_for("model.layers.0.self_attn.q_proj"),
        (64, 8, "affine".to_string())
    );
}

#[test]
fn config_validation_names_the_problem() {
    let mut config = checkpoint_config();
    config["num_key_value_heads"] = json!(3);
    let error = PrivacyFilterArgs::from_json(&config.to_string()).unwrap_err();
    assert!(error.contains("divisible"), "{error}");

    let mut config = checkpoint_config();
    config["rope_parameters"]["rope_type"] = json!("default");
    let error = PrivacyFilterArgs::from_json(&config.to_string()).unwrap_err();
    assert!(error.contains("requires YaRN"), "{error}");

    let mut config = checkpoint_config();
    config["attention_chunk_size"] = json!(0);
    assert!(
        PrivacyFilterArgs::from_json(&config.to_string())
            .unwrap_err()
            .contains("attention_chunk_size")
    );

    // Defaults cover a config that only names the family.
    let args = PrivacyFilterArgs::from_json(r#"{"model_type": "openai_privacy_filter"}"#).unwrap();
    assert_eq!(args.hidden_size, 640);
    assert_eq!(args.yarn().0.len(), 32);
}

/// Deterministic pseudo-random values in `[-scale, scale]`.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * scale
    }

    fn array(&mut self, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
        let count: i32 = shape.iter().product();
        let data: Vec<f32> = (0..count).map(|_| self.next_f32(scale)).collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    assert_eq!(mlxcel_core::array_shape(a), mlxcel_core::array_shape(b));
    let a = array_to_vec_f32(&mlxcel_core::astype(a, dtype::FLOAT32));
    let b = array_to_vec_f32(&mlxcel_core::astype(b, dtype::FLOAT32));
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn band_mask_is_the_symmetric_window() {
    let _guard = mlx_test_guard();
    let mask = band_mask(2, 5, 0, 7, 2);
    assert_eq!(mlxcel_core::array_shape(&mask), vec![1, 1, 3, 7]);
    let values = array_to_vec_f32(&mlxcel_core::astype(&mask, dtype::FLOAT32));
    // query 2 sees keys 0..=4, query 3 sees 1..=5, query 4 sees 2..=6.
    assert_eq!(
        values,
        [
            1., 1., 1., 1., 1., 0., 0., //
            0., 1., 1., 1., 1., 1., 0., //
            0., 0., 1., 1., 1., 1., 1.,
        ]
    );
}

/// 300 tokens through 64-query chunks equal one SDPA call under the full
/// `T x T` band mask, with GQA and sinks.
#[test]
fn banded_attention_matches_the_full_mask() {
    let _guard = mlx_test_guard();
    let (tokens, heads, kv_heads, head_dim, radius) = (300, 4, 2, 8, 16);
    let mut rng = Lcg(7);
    let q = rng.array(&[1, heads, tokens, head_dim], 1.0);
    let k = rng.array(&[1, kv_heads, tokens, head_dim], 1.0);
    let v = rng.array(&[1, kv_heads, tokens, head_dim], 1.0);
    let sinks = rng.array(&[heads], 2.0);

    let full_mask = band_mask(0, tokens, 0, tokens, radius);
    let mask_ptr = full_mask.as_ref().unwrap() as *const MlxArray;
    let sinks_ptr = sinks.as_ref().unwrap() as *const MlxArray;
    let reference = unsafe {
        mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
            &q, &k, &v, 1.0, mask_ptr, sinks_ptr,
        )
    };
    for chunk in [64, 128, 1024] {
        let chunked = banded_attention(&q, &k, &v, &sinks, radius, chunk);
        mlxcel_core::eval(&chunked);
        assert_eq!(
            mlxcel_core::array_shape(&chunked),
            vec![1, heads, tokens, head_dim]
        );
        let diff = max_abs_diff(&chunked, &reference);
        assert!(diff < 1e-4, "chunk {chunk}: max |diff| {diff}");
    }
}

fn tiny_config(attention_chunk: usize, moe_chunk: usize) -> Value {
    json!({
        "model_type": "openai_privacy_filter",
        "vocab_size": 64, "hidden_size": 16, "intermediate_size": 12,
        "num_hidden_layers": 2, "num_local_experts": 6, "num_experts_per_tok": 2,
        "head_dim": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
        "sliding_window": 3, "default_n_ctx": 512, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                            "beta_fast": 32.0, "beta_slow": 1.0, "truncate": false,
                            "original_max_position_embeddings": 64},
        "attention_chunk_size": attention_chunk, "moe_chunk_size": moe_chunk,
        "id2label": {"0": "O", "1": "B-x", "2": "I-x", "3": "E-x", "4": "S-x"},
    })
}

/// Random f32 weights for `tiny_config`, in the split mlx-community layout.
fn tiny_weights(args: &PrivacyFilterArgs, seed: u64) -> WeightMap {
    let mut rng = Lcg(seed);
    let (h, inter, e) = (
        args.hidden_size as i32,
        args.intermediate_size as i32,
        args.num_local_experts as i32,
    );
    let (q_out, kv_out) = (
        (args.num_attention_heads * args.head_dim) as i32,
        (args.num_key_value_heads * args.head_dim) as i32,
    );
    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".into(),
        rng.array(&[args.vocab_size as i32, h], 1.0),
    );
    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        for (name, out, inp) in [
            ("q_proj", q_out, h),
            ("k_proj", kv_out, h),
            ("v_proj", kv_out, h),
            ("o_proj", h, q_out),
        ] {
            weights.insert(
                format!("{p}.self_attn.{name}.weight"),
                rng.array(&[out, inp], 0.3),
            );
            weights.insert(format!("{p}.self_attn.{name}.bias"), rng.array(&[out], 0.1));
        }
        weights.insert(
            format!("{p}.self_attn.sinks"),
            rng.array(&[args.num_attention_heads as i32], 1.0),
        );
        weights.insert(format!("{p}.mlp.router.weight"), rng.array(&[e, h], 0.5));
        weights.insert(format!("{p}.mlp.router.bias"), rng.array(&[e], 0.1));
        for (name, out, inp) in [
            ("gate_proj", inter, h),
            ("up_proj", inter, h),
            ("down_proj", h, inter),
        ] {
            weights.insert(
                format!("{p}.mlp.experts.{name}.weight"),
                rng.array(&[e, out, inp], 0.3),
            );
            weights.insert(
                format!("{p}.mlp.experts.{name}.bias"),
                rng.array(&[e, out], 0.1),
            );
        }
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            weights.insert(format!("{p}.{norm}.weight"), rng.array(&[h], 1.0));
        }
    }
    weights.insert("model.norm.weight".into(), rng.array(&[h], 1.0));
    weights.insert("score.weight".into(), rng.array(&[5, h], 0.5));
    weights.insert("score.bias".into(), rng.array(&[5], 0.1));
    weights
}

/// A 40-token forward is the same whether attention runs in 7-query chunks
/// or one, and whether routing runs in 16-token chunks or one.
#[test]
fn tiny_model_logits_are_chunk_invariant() {
    let _guard = mlx_test_guard();
    let reference_args = args_from(&tiny_config(4096, 2048));
    let weights = tiny_weights(&reference_args, 11);
    let ids: Vec<i32> = (0..40).map(|i| (i * 7 + 3) % 64).collect();
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, 40]);

    let reference = PrivacyFilterModel::from_weights(&weights, reference_args).unwrap();
    assert_eq!(reference.num_labels(), 5);
    assert_eq!(reference.vocab_rows(), 64);
    let reference_logits = reference.logits(&input_ids);
    mlxcel_core::eval(&reference_logits);
    assert_eq!(mlxcel_core::array_shape(&reference_logits), vec![1, 40, 5]);
    let flat = array_to_vec_f32(&reference_logits);
    assert!(flat.iter().all(|v| v.is_finite()));

    for (attention_chunk, moe_chunk) in [(7, 2048), (4096, 16), (7, 16), (1, 1)] {
        let model = PrivacyFilterModel::from_weights(
            &weights,
            args_from(&tiny_config(attention_chunk, moe_chunk)),
        )
        .unwrap();
        let logits = model.logits(&input_ids);
        mlxcel_core::eval(&logits);
        let diff = max_abs_diff(&logits, &reference_logits);
        assert!(
            diff < 1e-3,
            "attention chunk {attention_chunk}, moe chunk {moe_chunk}: max |diff| {diff}"
        );
    }
}

#[test]
fn moe_block_runs_both_gather_paths_identically() {
    let _guard = mlx_test_guard();
    let args = args_from(&tiny_config(4096, 2048));
    let weights = tiny_weights(&args, 3);
    let moe = PrivacyFilterMoe::from_weights(&weights, &args, "model.layers.0.mlp").unwrap();
    let mut rng = Lcg(5);
    // 40 tokens x top-2 = 80 routed pairs in one chunk, or 32 per 16-token
    // chunk; dense planes gather unsorted either way.
    let x = rng.array(&[1, 40, 16], 1.0);
    let whole = moe.forward(&x);
    let chunked_args = args_from(&tiny_config(4096, 16));
    let chunked =
        PrivacyFilterMoe::from_weights(&weights, &chunked_args, "model.layers.0.mlp").unwrap();
    let in_chunks = chunked.forward(&x);
    mlxcel_core::eval(&whole);
    mlxcel_core::eval(&in_chunks);
    assert_eq!(mlxcel_core::array_shape(&whole), vec![1, 40, 16]);
    let diff = max_abs_diff(&whole, &in_chunks);
    assert!(diff < 1e-4, "max |diff| {diff}");
}

#[test]
fn sanitize_splits_the_fused_hugging_face_layout() {
    let _guard = mlx_test_guard();
    let (experts, hidden, inter) = (2, 3, 2);
    let fused: Vec<f32> = (0..(experts * hidden * 2 * inter))
        .map(|v| v as f32)
        .collect();
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.experts.gate_up_proj".into(),
        mlxcel_core::from_slice_f32(&fused, &[experts, hidden, 2 * inter]),
    );
    let bias: Vec<f32> = (0..(experts * 2 * inter))
        .map(|v| 100.0 + v as f32)
        .collect();
    weights.insert(
        "model.layers.0.mlp.experts.gate_up_proj_bias".into(),
        mlxcel_core::from_slice_f32(&bias, &[experts, 2 * inter]),
    );
    let down: Vec<f32> = (0..(experts * inter * hidden))
        .map(|v| 200.0 + v as f32)
        .collect();
    weights.insert(
        "model.layers.0.mlp.experts.down_proj".into(),
        mlxcel_core::from_slice_f32(&down, &[experts, inter, hidden]),
    );
    weights.insert(
        "model.layers.0.mlp.experts.down_proj_bias".into(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[experts, hidden]),
    );
    // The earlier fused MLX runtime layout, quantization companion included.
    weights.insert(
        "model.layers.1.mlp.experts.gate_up_proj.scales".into(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]),
    );
    sanitize_weights(&mut weights);

    let shape = |key: &str| mlxcel_core::array_shape(&weights[key]);
    let values = |key: &str| array_to_vec_f32(&weights[key]);
    assert!(!weights.contains_key("model.layers.0.mlp.experts.gate_up_proj"));
    assert_eq!(
        shape("model.layers.0.mlp.experts.gate_proj.weight"),
        vec![2, 2, 3]
    );
    assert_eq!(
        shape("model.layers.0.mlp.experts.up_proj.weight"),
        vec![2, 2, 3]
    );
    // Expert 0 of the fused tensor is [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    // ([hidden, 2 * inter]); transposed, its first two rows are the gate.
    assert_eq!(
        &values("model.layers.0.mlp.experts.gate_proj.weight")[..6],
        &[0., 4., 8., 1., 5., 9.]
    );
    assert_eq!(
        &values("model.layers.0.mlp.experts.up_proj.weight")[..6],
        &[2., 6., 10., 3., 7., 11.]
    );
    assert_eq!(
        values("model.layers.0.mlp.experts.gate_proj.bias"),
        [100., 101., 104., 105.]
    );
    assert_eq!(
        values("model.layers.0.mlp.experts.up_proj.bias"),
        [102., 103., 106., 107.]
    );
    assert_eq!(
        shape("model.layers.0.mlp.experts.down_proj.weight"),
        vec![2, 3, 2]
    );
    assert_eq!(
        &values("model.layers.0.mlp.experts.down_proj.weight")[..6],
        &[200., 203., 201., 204., 202., 205.]
    );
    assert_eq!(
        values("model.layers.0.mlp.experts.down_proj.bias"),
        [1., 2., 3., 4., 5., 6.]
    );
    assert_eq!(
        values("model.layers.1.mlp.experts.gate_proj.scales"),
        [1., 2.]
    );
    assert_eq!(
        values("model.layers.1.mlp.experts.up_proj.scales"),
        [3., 4.]
    );
}

/// `mlxcel generate` refuses the checkpoint with the token-classifier
/// message, and embedding detection does not claim it.
#[test]
fn generation_dispatch_refuses_and_embedding_detection_declines() {
    let config = checkpoint_config();
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let error = crate::models::get_model_type(dir.path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("openai_privacy_filter is a token classifier; use `mlxcel tag`"),
        "{error}"
    );
    assert!(!error.contains("Unsupported model type"), "{error}");
    assert_eq!(
        crate::models::detection::is_embedding_checkpoint(dir.path(), &config)
            .expect("no embedding error")
            .map(|t| format!("{t:?}")),
        None
    );
    assert!(!crate::models::detection::is_encoder_only_model_type(
        "openai_privacy_filter"
    ));
}
