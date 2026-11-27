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

//! ModelOpt `MIXED_PRECISION` detection and the MXFP8 layer repack (#3437).

use super::*;
use crate::models::gemma4::{is_modelopt_nvfp4, validate_quantization_scheme};
use serde_json::json;

/// The shape of `KRAFTON/Raon-1-mini-nvfp4`'s `quantization_config`.
fn raon_declaration() -> Value {
    json!({
        "quant_method": "modelopt_fp4", "quant_algo": "MIXED_PRECISION", "group_size": 16,
        "kv_cache_scheme": null, "ignore": ["model.layers.1.moe.gate"],
        "quantized_layers": {
            "model.layers.1.moe.experts.0.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
            "model.layers.0.self_attn.q_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            "lm_head": {"quant_algo": "MXFP8", "group_size": 32}
        }
    })
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlxcel_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn detector_accepts_the_raon_mixed_precision_declaration() {
    let qc = raon_declaration();
    assert!(is_mixed_precision_declaration(&qc));
    assert!(is_modelopt_nvfp4(&qc));
    assert_eq!(mixed_precision_mxfp8_layers(&qc).unwrap(), vec!["lm_head"]);
    let config = json!({"model_type": "raon1_mini", "quantization_config": qc});
    assert!(validate_quantization_scheme(&config).is_ok());
    let dir = temp_dir("mixed_precision_detect");
    assert!(crate::models::is_modelopt_nvfp4_text_checkpoint(
        Some(&config),
        &dir
    ));
    // `modelopt` spells the same declaration.
    let mut plain = raon_declaration();
    plain["quant_method"] = json!("modelopt");
    assert!(is_modelopt_nvfp4(&plain));
}

#[test]
fn detector_rejects_an_unknown_per_layer_algorithm_by_name() {
    let mut qc = raon_declaration();
    qc["quantized_layers"]["model.layers.0.self_attn.k_proj"] = json!({"quant_algo": "FP8"});
    assert!(!is_modelopt_nvfp4(&qc));
    let err = mixed_precision_mxfp8_layers(&qc).unwrap_err();
    assert!(
        err.contains("model.layers.0.self_attn.k_proj") && err.contains("\"FP8\""),
        "{err}"
    );
    let config = json!({"quantization_config": qc});
    assert!(validate_quantization_scheme(&config).is_err());
    let dir = temp_dir("mixed_precision_reject");
    assert!(!crate::models::is_modelopt_nvfp4_text_checkpoint(
        Some(&config),
        &dir
    ));

    // A wrong block size, a declaration with no NVFP4 layer, and one with no
    // per-layer map are not NVFP4 exports either.
    let mut wrong_group = raon_declaration();
    wrong_group["quantized_layers"]["lm_head"]["group_size"] = json!(16);
    assert!(
        mixed_precision_mxfp8_layers(&wrong_group)
            .unwrap_err()
            .contains("group_size")
    );
    let only_mxfp8 = json!({"quant_method": "modelopt_fp4", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"lm_head": {"quant_algo": "MXFP8"}}});
    assert!(!is_modelopt_nvfp4(&only_mxfp8));
    let no_layers = json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"});
    assert!(!is_mixed_precision_declaration(&no_layers));
    assert!(!is_modelopt_nvfp4(&no_layers));
}

fn byte_values(a: &mlxcel_core::MlxArray) -> Vec<u8> {
    let bytes = mlxcel_core::view(&mlxcel_core::contiguous(a, false), dtype::UINT8);
    mlxcel_core::eval(&bytes);
    mlxcel_core::array_to_raw_bytes(&bytes)
}

#[test]
fn mxfp8_layer_is_viewed_as_mlx_mxfp8_without_changing_a_byte() {
    let (rows, cols) = (3i32, 64i32);
    let codes: Vec<u8> = (0..rows * cols).map(|i| (i * 37 % 251) as u8).collect();
    let scales: Vec<u8> = (0..rows * cols / 32).map(|i| 120 + i as u8).collect();
    let mut weights = WeightMap::new();
    weights.insert(
        "lm_head.weight".into(),
        mlxcel_core::from_bytes(&codes, &[rows, cols], dtype::UINT8),
    );
    weights.insert(
        "lm_head.weight_scale".into(),
        mlxcel_core::from_bytes(&scales, &[rows, cols / 32], dtype::UINT8),
    );
    weights.insert(
        "lm_head.input_scale".into(),
        mlxcel_core::from_slice_f32(&[1.0], &[]),
    );
    // Another family's loader would read the plane as MXFP4 g64, so its
    // MXFP8 layer is refused by name and the map is left as it was.
    let other = json!({"model_type": "llama", "quantization_config": raon_declaration()});
    let err = repack_mixed_precision_mxfp8_layers(&mut weights, Some(&other)).unwrap_err();
    assert!(err.contains("lm_head is MXFP8"), "{err}");
    assert_eq!(array_dtype(&weights["lm_head.weight"]), dtype::UINT8);

    let config = json!({"model_type": "raon1_mini", "quantization_config": raon_declaration()});
    repack_mixed_precision_mxfp8_layers(&mut weights, Some(&config)).unwrap();

    let w = &weights["lm_head.weight"];
    assert_eq!(array_dtype(w), dtype::UINT32);
    assert_eq!(array_shape(w), vec![rows, cols / 4]);
    assert_eq!(byte_values(w), codes);
    assert_eq!(byte_values(&weights["lm_head.scales"]), scales);
    assert!(!weights.contains_key("lm_head.weight_scale"));
    assert!(!weights.contains_key("lm_head.input_scale"));

    // A widened (non-uint8) head is refused by name, and a config without a
    // mixed-precision declaration leaves the map alone.
    let mut widened = WeightMap::new();
    widened.insert(
        "lm_head.weight".into(),
        mlxcel_core::zeros(&[rows, cols], dtype::FLOAT16),
    );
    widened.insert(
        "lm_head.weight_scale".into(),
        mlxcel_core::zeros(&[rows, cols / 32], dtype::UINT8),
    );
    let err = repack_mixed_precision_mxfp8_layers(&mut widened, Some(&config)).unwrap_err();
    assert!(err.contains("uint8"), "{err}");
    let mut untouched = WeightMap::new();
    untouched.insert(
        "lm_head.weight".into(),
        mlxcel_core::zeros(&[rows, cols], dtype::UINT8),
    );
    repack_mixed_precision_mxfp8_layers(&mut untouched, Some(&json!({}))).unwrap();
    assert_eq!(array_dtype(&untouched["lm_head.weight"]), dtype::UINT8);
}
