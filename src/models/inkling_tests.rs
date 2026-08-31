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

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, dtype};
use serde_json::json;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::InklingConfig;
use super::attention::{InklingShortConv, banded_additive_mask, log_scaling_tau};
use super::mlp::route_weights;
use super::sanitize::sanitize_weights;
use super::validation::validate_config;

fn config(text: serde_json::Value) -> InklingConfig {
    serde_json::from_value(json!({"text_config": text})).unwrap()
}

fn f32_values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn nested_hf_nvfp4_sidecar_promotes_null_quantization() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mlxcel_inkling_nvfp4_{nonce}"));
    fs::create_dir_all(&path).unwrap();
    fs::write(
        path.join("hf_quant_config.json"),
        serde_json::to_vec(&json!({
            "quantization": {"quant_algo": "NVFP4", "group_size": 16}
        }))
        .unwrap(),
    )
    .unwrap();
    let config = InklingConfig::from_json_with_sidecar(
        &path,
        &json!({"text_config": {}, "quantization": null}).to_string(),
    )
    .unwrap();
    assert_eq!(config.quantization(), (16, 4, "nvfp4"));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn eos_defaults_and_scalar_override_are_preserved() {
    let defaulted = config(json!({"intermediate_size": 32, "dense_intermediate_size": 64}));
    assert_eq!(defaulted.eos_token_ids(), [200_006]);
    let overridden: InklingConfig = serde_json::from_value(json!({
        "text_config": {"intermediate_size": 32, "dense_intermediate_size": 64},
        "eos_token_id": 42
    }))
    .unwrap();
    assert_eq!(overridden.eos_token_ids(), [42]);
}

#[test]
fn layer_classification_layer_types_then_local_ids_then_modulo() {
    let typed = config(json!({
        "num_hidden_layers": 4,
        "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        "mlp_layer_types": ["dense", "sparse", "dense", "sparse"],
        "intermediate_size": 32,
        "dense_intermediate_size": 64
    }));
    assert_eq!(
        (0..4)
            .filter(|&i| typed.text_config.layer_is_sliding(i))
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(
        (0..4)
            .filter(|&i| typed.text_config.layer_is_dense(i))
            .collect::<Vec<_>>(),
        [0, 2]
    );

    let listed = config(json!({
        "num_hidden_layers": 8,
        "local_layer_ids": [0, 2, 7],
        "dense_mlp_idx": 2,
        "intermediate_size": 32,
        "dense_intermediate_size": 64
    }));
    assert_eq!(
        (0..8)
            .filter(|&i| listed.text_config.layer_is_sliding(i))
            .collect::<Vec<_>>(),
        [0, 2, 7]
    );
    assert_eq!(
        (0..8)
            .filter(|&i| listed.text_config.layer_is_dense(i))
            .collect::<Vec<_>>(),
        [0, 1]
    );

    let defaulted = config(json!({
        "num_hidden_layers": 12,
        "intermediate_size": 32,
        "dense_intermediate_size": 64
    }));
    assert_eq!(
        (0..12)
            .filter(|&i| !defaulted.text_config.layer_is_sliding(i))
            .collect::<Vec<_>>(),
        [5, 11]
    );
}

#[test]
fn validation_rejects_unknown_layer_schedule_entries() {
    let bad_attention = config(json!({
        "num_hidden_layers": 1,
        "layer_types": ["hybrid_typo"],
        "intermediate_size": 32,
        "dense_intermediate_size": 64
    }));
    assert!(
        validate_config(&bad_attention)
            .unwrap_err()
            .contains("layer_types entries")
    );

    let bad_mlp = config(json!({
        "num_hidden_layers": 1,
        "mlp_layer_types": ["moe"],
        "intermediate_size": 32,
        "dense_intermediate_size": 64
    }));
    assert!(
        validate_config(&bad_mlp)
            .unwrap_err()
            .contains("mlp_layer_types entries")
    );
}

#[test]
fn mlp_width_resolution_accepts_both_spellings() {
    let hf = config(json!({"intermediate_size": 4096, "moe_intermediate_size": 512}));
    assert_eq!(hf.text_config.widths().unwrap(), (4096, 512));
    let native = config(json!({"intermediate_size": 2048, "dense_intermediate_size": 16384}));
    assert_eq!(native.text_config.widths().unwrap(), (16384, 2048));
}

#[test]
fn banded_bias_matches_reference_formula() {
    let (batch, length, heads, d_rel, source, extent, window) = (1, 5, 2, 3, 9, 6, 4);
    let r_data: Vec<f32> = (0..batch * length * heads * d_rel)
        .map(|i| (i as f32 - 7.0) / 13.0)
        .collect();
    let p_data: Vec<f32> = (0..d_rel * extent)
        .map(|i| (i as f32 + 1.0) / 17.0)
        .collect();
    let r = mlxcel_core::from_slice_f32(&r_data, &[batch, length, heads, d_rel]);
    let p = mlxcel_core::from_slice_f32(&p_data, &[d_rel, extent]);
    let actual = banded_additive_mask(&r, &p, 4, source, Some(window), extent);
    let actual = f32_values(&actual);
    for h in 0..heads {
        for i in 0..length {
            for j in 0..source {
                let dist = i + 4 - j;
                let index = ((h * length + i) * source + j) as usize;
                if dist < 0 || dist >= window {
                    assert!(actual[index] < -1e29);
                } else if dist >= extent {
                    assert_eq!(actual[index], 0.0);
                } else {
                    let expected: f32 = (0..d_rel)
                        .map(|d| {
                            r_data[((i * heads + h) * d_rel + d) as usize]
                                * p_data[(d * extent + dist) as usize]
                        })
                        .sum();
                    assert!((actual[index] - expected).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn log_scaling_tau_is_identity_below_floor() {
    let tau = log_scaling_tau(9, 0, 8.0, 0.1);
    let values = f32_values(&tau);
    assert!(values[..8].iter().all(|value| (*value - 1.0).abs() < 1e-6));
    assert!((values[8] - (1.0 + 0.1 * (9.0_f32 / 8.0).ln())).abs() < 1e-6);
}

#[test]
fn sconv_prefill_continuation_matches_one_shot_and_state() {
    let mut weights = WeightMap::new();
    let weight =
        mlxcel_core::from_slice_f32(&[0.1, 0.2, -0.3, 0.4, -0.2, 0.5, 0.25, -0.1], &[2, 4, 1]);
    weights.insert("conv.weight".into(), weight);
    let conv = InklingShortConv::from_weights(&weights, "conv.weight", 4).unwrap();
    let input: Vec<f32> = (0..20).map(|i| (i as f32 - 5.0) / 11.0).collect();
    let input = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&input, &[1, 10, 2]),
        dtype::BFLOAT16,
    );
    let mut full_state = None;
    let full = conv.forward(&input, &mut full_state, None);

    let first = mlxcel_core::utils::slice_axis(&input, 1, 0, 7);
    let mut resumed_state = None;
    let mut resumed = conv.forward(&first, &mut resumed_state, None);
    for pos in 7..10 {
        let token = mlxcel_core::utils::slice_axis(&input, 1, pos, pos + 1);
        let out = conv.forward(&token, &mut resumed_state, None);
        resumed = mlxcel_core::concatenate(&resumed, &out, 1);
    }
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full, &resumed, 1e-3, 1e-3
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        full_state.as_deref().unwrap(),
        resumed_state.as_deref().unwrap(),
        1e-6,
        1e-6,
    )));
}

#[test]
fn router_normalizes_selected_and_shared_raw_logits_together() {
    let raw_logits = [0.1_f32, 0.2, 2.0, -1.0, 0.3, -0.4];
    let logits = mlxcel_core::from_slice_f32(&raw_logits, &[1, 6]);
    let correction = mlxcel_core::from_slice_f32(&[5.0, 0.0, 0.0, 0.0], &[4]);
    let global = mlxcel_core::from_slice_f32(&[2.0], &[1]);
    let (indices, routed, shared) = route_weights(&logits, &correction, &global, 4, 2, 2, 8.0);
    let index_bytes = mlxcel_core::array_to_raw_bytes(&indices);
    let indices: Vec<u32> = index_bytes
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    assert!(
        indices.contains(&0),
        "correction bias must affect selection"
    );
    let actual: Vec<f32> = f32_values(&routed)
        .into_iter()
        .chain(f32_values(&shared))
        .collect();
    let selected_and_shared = [
        raw_logits[indices[0] as usize],
        raw_logits[indices[1] as usize],
        raw_logits[4],
        raw_logits[5],
    ];
    let sigmoid = selected_and_shared.map(|value| 1.0 / (1.0 + (-value).exp()));
    let denominator: f32 = sigmoid.iter().sum();
    let expected = sigmoid.map(|value| value / denominator * 16.0);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5);
    }
    let sum: f32 = actual.iter().sum();
    assert!((sum - 16.0).abs() < 1e-5);
}

#[test]
fn sanitize_original_layout_is_idempotent_and_drops_towers() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.llm.embed.weight".into(),
        mlxcel_core::zeros(&[8, 4], dtype::FLOAT32),
    );
    weights.insert(
        "model.llm.layers.0.attn.k_sconv.weight".into(),
        mlxcel_core::zeros(&[4, 1, 3], dtype::FLOAT32),
    );
    weights.insert(
        "model.llm.layers.0.mlp.w13_dn.weight".into(),
        mlxcel_core::zeros(&[6, 4], dtype::FLOAT32),
    );
    weights.insert(
        "model.visual.fake".into(),
        mlxcel_core::zeros(&[1], dtype::FLOAT32),
    );
    let once = sanitize_weights(weights).unwrap();
    assert!(once.contains_key("model.embed_tokens.weight"));
    assert_eq!(
        mlxcel_core::array_shape(&once["model.layers.0.self_attn.k_sconv.conv.weight"]),
        [4, 3, 1]
    );
    assert!(once.contains_key("model.layers.0.mlp.gate_proj.weight"));
    assert!(!once.keys().any(|key| key.starts_with("model.visual")));
    let mut keys = once.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    let twice = sanitize_weights(once).unwrap();
    let mut twice_keys = twice.keys().cloned().collect::<Vec<_>>();
    twice_keys.sort();
    assert_eq!(keys, twice_keys);
}

#[test]
fn sanitize_native_nvfp4_preserves_packing_and_expert_scales() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.llm.layers.0.mlp.experts.w13_weight".into(),
        mlxcel_core::zeros(&[2, 4, 8], dtype::UINT8),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w2_weight".into(),
        mlxcel_core::zeros(&[2, 4, 8], dtype::UINT8),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w13_weight.scale".into(),
        mlxcel_core::ones(&[2, 4, 1], dtype::FLOAT32),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w2_weight.scale".into(),
        mlxcel_core::ones(&[2, 4, 1], dtype::FLOAT32),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w13_weight.scale2".into(),
        mlxcel_core::from_slice_f32(&[2.0, 3.0], &[2]),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w2_weight.scale2".into(),
        mlxcel_core::from_slice_f32(&[5.0, 7.0], &[2]),
    );
    let sanitized = sanitize_weights(weights).unwrap();
    let prefix = "model.layers.0.mlp.switch_mlp";
    assert_eq!(
        mlxcel_core::array_shape(&sanitized[&format!("{prefix}.gate_proj.weight")]),
        [2, 2, 2]
    );
    assert_eq!(
        mlxcel_core::array_dtype(&sanitized[&format!("{prefix}.gate_proj.scales")]),
        dtype::UINT8
    );
    assert_eq!(
        f32_values(&sanitized[&format!("{prefix}.gate_scale")]),
        [2.0, 3.0]
    );
    assert_eq!(
        f32_values(&sanitized[&format!("{prefix}.out_scale")]),
        [10.0, 21.0]
    );
}

#[test]
fn sanitize_preconverted_affine_experts_keep_sidecars_and_merge_shared_rows() {
    let mut weights = WeightMap::new();
    for leaf in ["weight", "scales", "biases"] {
        weights.insert(
            format!("model.llm.layers.0.mlp.experts.gate_proj.{leaf}"),
            mlxcel_core::zeros(&[2, 2, 1], dtype::UINT32),
        );
        weights.insert(
            format!("model.llm.layers.0.mlp.shared_experts.gate_proj.{leaf}"),
            mlxcel_core::zeros(&[2, 2, 1], dtype::UINT32),
        );
    }
    let sanitized = sanitize_weights(weights).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&sanitized["model.layers.0.mlp.switch_mlp.gate_proj.scales"]),
        [2, 2, 1]
    );
    assert_eq!(
        mlxcel_core::array_shape(&sanitized["model.layers.0.mlp.shared_experts.gate_proj.biases"]),
        [4, 1]
    );
}

#[test]
fn sanitize_rejects_incomplete_nvfp4_experts() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.llm.layers.0.mlp.experts.w13_weight".into(),
        mlxcel_core::zeros(&[2, 4, 4], dtype::UINT8),
    );
    let error = match sanitize_weights(weights) {
        Ok(_) => panic!("incomplete NVFP4 experts must fail sanitization"),
        Err(error) => error,
    };
    assert!(error.contains("missing routed w2"));
}

#[test]
fn sanitize_rejects_malformed_nvfp4_before_reinterpretation() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.llm.layers.0.mlp.experts.w13_weight".into(),
        mlxcel_core::zeros(&[2, 4, 6], dtype::UINT8),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w2_weight".into(),
        mlxcel_core::zeros(&[2, 4, 6], dtype::UINT8),
    );
    let error = match sanitize_weights(weights) {
        Ok(_) => panic!("misaligned NVFP4 weights must fail sanitization"),
        Err(error) => error,
    };
    assert!(error.contains("divisible by 4"));
}

#[test]
fn sanitize_rejects_reverse_mixed_routed_weight_dtypes() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.llm.layers.0.mlp.experts.w13_weight".into(),
        mlxcel_core::zeros(&[2, 4, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.llm.layers.0.mlp.experts.w2_weight".into(),
        mlxcel_core::zeros(&[2, 4, 8], dtype::UINT8),
    );
    let error = match sanitize_weights(weights) {
        Ok(_) => panic!("mixed routed expert storage must fail sanitization"),
        Err(error) => error,
    };
    assert!(error.contains("cannot mix native NVFP4"));
}
