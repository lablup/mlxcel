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

// The literals are upstream's f32 values printed at full f64 precision.
#![allow(clippy::excessive_precision)]

use mlxcel_core::weights::WeightMap;

use super::*;
use crate::imagegen::testing::values;

fn u8_tensor(bytes: &[u8], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_bytes(bytes, shape, dtype::UINT8)
}

/// Each row mixes normals, a negative, zero, the 448 maximum and the
/// subnormals `0x01` (2^-9) and `0x81` (-2^-9).
const RAW: [u8; 32] = [
    0x38, 0x40, 0xB8, 0x01, 0x7E, 0x00, 0x30, 0x3C, //
    0x81, 0x07, 0x08, 0x3F, 0xC0, 0x50, 0x2A, 0x39, //
    0x3A, 0x44, 0xF0, 0x12, 0x5B, 0x8F, 0x37, 0x01, //
    0x46, 0xA3, 0x3B, 0x0D, 0x60, 0xEE, 0x48, 0x05,
];
const SCALES: [f32; 4] = [1.0, 0.5, 3.0, 0.1];

/// Upstream `dequantize_fp8_weight_only` on the same tensor (mlx 0.32.4,
/// `mx.from_fp8(bytes, bf16) * scale.astype(bf16)[:, None]`). Row 0 is the
/// E4M3 decode itself (scale 1); row 3's 0.1 scale is rounded to bf16
/// (0.10009765625) before the product, which is rounded to bf16 again
/// (`0x46` = 3.5 gives 0.349609375, not 0.35).
const WANT: [f32; 32] = [
    1.0,
    2.0,
    -1.0,
    0.001953125,
    448.0,
    0.0,
    0.5,
    1.5, //
    -0.0009765625,
    0.0068359375,
    0.0078125,
    0.9375,
    -1.0,
    4.0,
    0.15625,
    0.5625, //
    3.75,
    9.0,
    -384.0,
    0.1171875,
    66.0,
    -0.087890625,
    2.8125,
    0.005859375, //
    0.349609375,
    -0.0172119140625,
    0.1376953125,
    0.0025482177734375,
    3.203125,
    -11.1875,
    0.400390625,
    0.0009765625,
];

#[test]
fn rowwise_fp8_matches_upstream_on_a_4x8_tensor() {
    let mut w = WeightMap::new();
    w.insert("a.weight".into(), u8_tensor(&RAW, &[4, 8]));
    w.insert(
        "a.weight_scale".into(),
        mlxcel_core::from_slice_f32(&SCALES, &[4]),
    );
    w.insert(
        "a.bias".into(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
    );
    // An unpaired scale is dropped; integer tensors pass through.
    w.insert(
        "b.weight_scale".into(),
        mlxcel_core::from_slice_f32(&[2.0], &[1]),
    );
    w.insert("c.idx".into(), mlxcel_core::from_slice_i32(&[1, 2], &[2]));
    let out = dequantize_fp8_rowwise(w).unwrap();
    let mut keys: Vec<&String> = out.keys().collect();
    keys.sort();
    assert_eq!(keys, ["a.bias", "a.weight", "c.idx"]);
    let weight = &out["a.weight"];
    assert_eq!(mlxcel_core::array_dtype(weight), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(weight), [4, 8]);
    assert_eq!(values(weight), WANT);
    assert_eq!(mlxcel_core::array_dtype(&out["a.bias"]), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_dtype(&out["c.idx"]), dtype::INT32);
}

#[test]
fn a_scale_that_is_not_per_row_names_the_tensor() {
    let mut w = WeightMap::new();
    w.insert(
        "layers.0.attention.qkv.weight".into(),
        u8_tensor(&RAW, &[4, 8]),
    );
    w.insert(
        "layers.0.attention.qkv.weight_scale".into(),
        mlxcel_core::from_slice_f32(&[1.0; 8], &[8]),
    );
    let Err(err) = dequantize_fp8_rowwise(w) else {
        panic!("expected an error")
    };
    let err = err.to_string();
    assert!(err.contains("layers.0.attention.qkv.weight"), "{err}");
    assert!(err.contains("[8]") && err.contains("[4]"), "{err}");
}

#[test]
fn a_scaled_weight_must_be_raw_fp8_bytes() {
    let mut w = WeightMap::new();
    w.insert(
        "x.weight".into(),
        mlxcel_core::from_slice_f32(&[1.0; 4], &[2, 2]),
    );
    w.insert(
        "x.weight_scale".into(),
        mlxcel_core::from_slice_f32(&[1.0; 2], &[2]),
    );
    let Err(err) = dequantize_fp8_rowwise(w) else {
        panic!("expected an error")
    };
    let err = err.to_string();
    assert!(err.contains("x.weight") && err.contains("uint8"), "{err}");
}

#[test]
fn the_text_encoder_keeps_only_the_language_model() {
    assert!(is_text_key("language_model.layers.0.mlp.up_proj.weight"));
    assert!(is_text_key(
        "language_model.layers.0.mlp.up_proj.weight_scale"
    ));
    assert!(is_text_key("language_model.embed_tokens.weight"));
    assert!(is_text_key("language_model.norm.weight"));
    assert!(!is_text_key("visual.blocks.0.attn.qkv.weight"));
    assert!(!is_text_key("visual.merger.norm.weight"));
    assert!(!is_text_key("lm_head.weight"));
}

#[test]
fn the_unconditional_branch_skips_only_the_llm_projection() {
    for key in ["llm_cond_norm.weight", "llm_cond_proj.weight"] {
        assert!(keeps_transformer_key(true, key), "{key}");
        assert!(!keeps_transformer_key(false, key), "{key}");
    }
    for key in ["blocks.0.attn.qkv.weight", "final_layer.weight"] {
        assert!(keeps_transformer_key(true, key), "{key}");
        assert!(keeps_transformer_key(false, key), "{key}");
    }
}

#[test]
fn the_text_config_resolves_rope_theta_from_rope_parameters() {
    // The checkpoint's `text_config` spells theta only inside the
    // transformers 5 `rope_parameters` block.
    let text = r#"{"text_config": {"attention_bias": false, "head_dim": 128, "hidden_size": 4096,
        "intermediate_size": 12288, "max_position_embeddings": 262144, "model_type": "qwen3_vl_text",
        "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06, "vocab_size": 151936,
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [24, 20, 20],
                            "rope_theta": 5000000, "rope_type": "default"}}}"#;
    let config: serde_json::Value = serde_json::from_str(text).unwrap();
    let args: crate::models::qwen3::ModelArgs =
        serde_json::from_value(config["text_config"].clone()).unwrap();
    assert_eq!(args.rope_theta, 5_000_000.0);
    assert_eq!(
        crate::models::rope_overrides::resolve_base(args.rope_theta),
        5_000_000.0
    );
}
