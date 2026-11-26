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

//! Raon-1 mini config rules: the published layer lists and the named
//! refusals.

use super::{Raon1MiniConfig, validate_layer_partition};

#[test]
fn published_layer_lists_map_to_the_zero_based_mla_layers() {
    let json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/models/raon1_mini_config_fixture.json"
    ))
    .expect("fixture");
    let config = Raon1MiniConfig::from_json_str(&json).expect("published config parses");
    assert_eq!(config.mla_layer_indices(), vec![3, 7, 11, 15, 19, 23, 24]);
    assert!(!config.is_moe_layer(0));
    assert!((1..25).all(|i| config.is_moe_layer(i)));
    assert_eq!(config.eos_token_ids(), vec![259202, 259200]);

    let rda = config.rda_config.rda_layers.clone();
    let mla = config.rda_config.mla_layers.clone();
    assert!(validate_layer_partition(&rda, &mla, 25).is_ok());
    let missing = &mla[..mla.len() - 1];
    assert!(
        validate_layer_partition(&rda, missing, 25)
            .unwrap_err()
            .contains("cover layer 25")
    );
    let mut dup = mla.clone();
    dup.push(1);
    assert!(
        validate_layer_partition(&rda, &dup, 25)
            .unwrap_err()
            .contains("twice")
    );
    let rda0: Vec<usize> = rda.iter().map(|l| l - 1).collect();
    let mla0: Vec<usize> = mla.iter().map(|l| l - 1).collect();
    assert!(
        validate_layer_partition(&rda0, &mla0, 25)
            .unwrap_err()
            .contains("1-based")
    );
}

#[test]
fn unsupported_configs_fail_by_name() {
    let base: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/models/raon1_mini_config_fixture.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let cases: [(&str, serde_json::Value, &str); 10] = [
        (
            "quantization",
            serde_json::json!({"group_size": 64, "bits": 4}),
            "nvfp4 with group_size 16",
        ),
        (
            "quantization",
            serde_json::json!({"group_size": 16, "bits": 4, "mode": "nvfp4",
                "lm_head": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}),
            "MLX quantization entry lm_head",
        ),
        (
            "quantization_config",
            serde_json::json!({"quant_method": "modelopt_fp4", "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"lm_head": {"quant_algo": "FP8"},
                    "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4", "group_size": 16}}}),
            "layer lm_head uses quant_algo \"FP8\"",
        ),
        ("hidden_act", serde_json::json!("silu"), "hidden_act"),
        (
            "moe_router_activation_func",
            serde_json::json!("softmax"),
            "sigmoid",
        ),
        (
            "use_grouped_topk",
            serde_json::json!(true),
            "use_grouped_topk",
        ),
        ("mla_use_nope", serde_json::json!(false), "mla_use_nope"),
        (
            "attn_res_block_size",
            serde_json::json!(6),
            "attn_res_block_size",
        ),
        ("raon_mhc_streams", serde_json::json!(1), "raon_mhc_streams"),
        (
            "quantization_config",
            serde_json::json!({"quant_method": "modelopt"}),
            "modelopt",
        ),
    ];
    for (key, value, needle) in cases {
        let mut config = base.clone();
        config[key] = value;
        let err = Raon1MiniConfig::from_json_str(&config.to_string()).unwrap_err();
        assert!(err.contains(needle), "{key}: {err}");
    }
}

/// `config.json` is untrusted: sizes that would truncate in an `as i32`
/// cast, overflow a shape product, or size an allocation from the config
/// alone fail by name before any weight is read.
#[test]
fn oversized_or_zero_geometry_fails_by_name() {
    let base: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/models/raon1_mini_config_fixture.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let huge = serde_json::json!(u64::MAX);
    let cases: [(&[&str], serde_json::Value, &str); 7] = [
        (&["num_hidden_layers"], huge.clone(), "cover layer 26"),
        (
            &["raon_mhc_streams"],
            serde_json::json!(100_000),
            "raon_mhc_streams",
        ),
        (
            &["num_attention_heads"],
            serde_json::json!(1u64 << 40),
            "overflows",
        ),
        (&["vocab_size"], serde_json::json!(1u64 << 32), "vocab_size"),
        (
            &["rda_config", "head_dim"],
            serde_json::json!(0),
            "rda_config.head_dim",
        ),
        (&["kv_lora_rank"], serde_json::json!(0), "kv_lora_rank"),
        (&["rda_config", "num_heads"], huge, "overflows"),
    ];
    for (path, value, needle) in cases {
        let mut config = base.clone();
        let mut slot = &mut config;
        for key in path {
            slot = &mut slot[*key];
        }
        *slot = value;
        let err = Raon1MiniConfig::from_json_str(&config.to_string()).unwrap_err();
        assert!(err.contains(needle), "{path:?}: {err}");
    }

    // A layer count far above the listed entries is refused by the
    // partition check without sizing anything from the count.
    assert!(
        validate_layer_partition(&[1, 2], &[3], usize::MAX)
            .unwrap_err()
            .contains("cover layer 4")
    );
}

/// The configs of both published NVFP4 exports parse (#3437): the vendor
/// MLX export with its runtime switches and `quantization` overrides, and the
/// ModelOpt mixed-precision export. Each gives its linears the NVFP4 layout.
#[test]
fn published_nvfp4_export_configs_parse() {
    let base: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/models/raon1_mini_config_fixture.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let original = Raon1MiniConfig::from_json_str(&base.to_string()).unwrap();
    assert_eq!(original.dense_quant(), (64, 4));

    let mut mlx = base.clone();
    let object = mlx.as_object_mut().unwrap();
    object.remove("quantization_config");
    object.insert(
        "quantization".into(),
        serde_json::json!({
            "group_size": 16, "bits": 4, "mode": "nvfp4",
            "model.layers.1.moe.switch_mlp.gate_proj":
                {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true},
            "lm_head": {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        }),
    );
    for (key, value) in [
        (
            "raon_global_scaled_linears",
            serde_json::json!(["model.layers.0.mlp.up_proj"]),
        ),
        ("raon_merged_gate_up", serde_json::json!(true)),
        ("raon_fused_mhc_decode", serde_json::json!(true)),
        ("raon_fp32_export_head", serde_json::json!(false)),
        ("model_file", serde_json::json!("raon_mlx_model.py")),
    ] {
        object.insert(key.into(), value);
    }
    let config = Raon1MiniConfig::from_json_str(&mlx.to_string()).expect("MLX export parses");
    assert_eq!(config.dense_quant(), (16, 4));

    let mut modelopt = base.clone();
    modelopt["quantization_config"] = serde_json::json!({
        "quant_method": "modelopt_fp4", "quant_algo": "MIXED_PRECISION", "group_size": 16,
        "kv_cache_scheme": null, "ignore": ["model.layers.1.moe.gate"],
        "quantized_layers": {
            "model.layers.1.moe.experts.0.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
            "model.layers.0.self_attn.q_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            "lm_head": {"quant_algo": "MXFP8", "group_size": 32}
        }
    });
    let config =
        Raon1MiniConfig::from_json_str(&modelopt.to_string()).expect("ModelOpt export parses");
    assert_eq!(config.dense_quant(), (16, 4));

    modelopt["quantization_config"]["kv_cache_scheme"] = serde_json::json!({"type": "float"});
    let err = Raon1MiniConfig::from_json_str(&modelopt.to_string()).unwrap_err();
    assert!(err.contains("kv_cache_scheme"), "{err}");
}
