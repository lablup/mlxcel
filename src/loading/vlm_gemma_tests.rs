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

use super::{
    gemma3n_language_model_prefix, gemma3n_metadata, gemma3n_needs_conv_transpose,
    parse_gemma3_vlm_config, prepare_gemma3_vlm_weights, sanitize_gemma3n_weights,
    sanitize_gemma4_audio_weights,
};
use crate::vision::config::{VLMConfig, VisionHiddenActivation};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

/// The fields of `mlx-community/gemma-3-4b-it-4bit`'s `config.json` that
/// `VLMConfig` reads, with the `vision_config` as shipped: no `hidden_act`.
fn gemma3_vlm_config() -> serde_json::Value {
    json!({
        "model_type": "gemma3",
        "boi_token_index": 255_999,
        "eoi_token_index": 256_000,
        "image_token_index": 262_144,
        "mm_tokens_per_image": 256,
        "text_config": {"model_type": "gemma3_text", "hidden_size": 2560},
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "image_size": 896,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_size": 14,
            "skip_vision": true,
            "vision_use_head": false
        }
    })
}

#[test]
fn gemma3_vlm_config_resolves_an_omitted_hidden_act_to_the_siglip_tanh_gelu() {
    // As shipped the key is absent, and the serde default alone is exact erf.
    // That is the tower the pinned eager-MLX-versus-IREE gate used to build
    // for its reference while the loader and the IREE graph ran tanh.
    let shipped = gemma3_vlm_config();
    let direct: VLMConfig = serde_json::from_value(shipped.clone()).unwrap();
    assert_eq!(
        direct.vision_config.hidden_act,
        VisionHiddenActivation::ExactGelu
    );
    let parsed = parse_gemma3_vlm_config(&shipped).unwrap();
    assert_eq!(
        parsed.vision_config.hidden_act,
        VisionHiddenActivation::GeluPytorchTanh
    );

    // A null key is an omitted key.
    let mut null_act = gemma3_vlm_config();
    null_act["vision_config"]["hidden_act"] = serde_json::Value::Null;
    assert_eq!(
        parse_gemma3_vlm_config(&null_act)
            .unwrap()
            .vision_config
            .hidden_act,
        VisionHiddenActivation::GeluPytorchTanh
    );

    // Nothing else of the config moves.
    assert_eq!(parsed.model_type, "gemma3");
    assert_eq!(parsed.image_token_index, 262_144);
    assert_eq!(parsed.boi_token_index, 255_999);
    assert_eq!(parsed.eoi_token_index, 256_000);
    assert_eq!(parsed.get_mm_tokens_per_image(), 256);
    assert_eq!(parsed.vision_config.num_hidden_layers, 27);
    assert_eq!(parsed.vision_config.hidden_size, 1152);
    assert_eq!(parsed.vision_config.intermediate_size, 4304);
    assert_eq!(parsed.vision_config.image_size, 896);
    assert_eq!(parsed.vision_config.patch_size, 14);
}

#[test]
fn gemma3_vlm_config_keeps_a_declared_hidden_act_and_the_loader_errors() {
    for (declared, expected) in [
        ("gelu", VisionHiddenActivation::ExactGelu),
        ("gelu_pytorch_tanh", VisionHiddenActivation::GeluPytorchTanh),
        ("quick_gelu", VisionHiddenActivation::QuickGelu),
    ] {
        let mut config = gemma3_vlm_config();
        config["vision_config"]["hidden_act"] = declared.into();
        assert_eq!(
            parse_gemma3_vlm_config(&config)
                .unwrap()
                .vision_config
                .hidden_act,
            expected,
            "{declared}"
        );
    }

    // A name that is not a GELU is still an error, with the loader's prefix.
    let mut not_a_gelu = gemma3_vlm_config();
    not_a_gelu["vision_config"]["hidden_act"] = "relu".into();
    let error = parse_gemma3_vlm_config(&not_a_gelu)
        .expect_err("relu")
        .to_string();
    assert!(error.starts_with("Failed to parse VLM config: "), "{error}");
    assert!(error.contains("unsupported vision hidden_act"), "{error}");

    let mut no_tower = gemma3_vlm_config();
    no_tower.as_object_mut().unwrap().remove("vision_config");
    let error = parse_gemma3_vlm_config(&no_tower)
        .expect_err("no vision_config")
        .to_string();
    assert!(error.starts_with("Failed to parse VLM config: "), "{error}");
}

#[test]
fn gemma3n_metadata_applies_defaults_and_overrides() {
    let defaults = gemma3n_metadata(&json!({}));
    assert_eq!(defaults.vision_hidden_size, 2048);
    assert_eq!(defaults.image_size, 256);
    assert_eq!(defaults.image_token_id, 262_145);
    assert_eq!(defaults.boi_token_id, 255_999);
    assert_eq!(defaults.eoi_token_id, 262_144);
    assert!((defaults.vision_rms_eps - 1e-6).abs() < f32::EPSILON);
    assert_eq!(defaults.audio_token_id, 262_273);
    assert_eq!(defaults.boa_token_id, 256_000);
    assert_eq!(defaults.eoa_token_id, 262_272);
    assert_eq!(defaults.audio_soft_tokens_per_clip, 188);

    let overrides = gemma3n_metadata(&json!({
        "vision_config": {
            "hidden_size": 3072,
            "image_size": 384,
            "rms_norm_eps": 1e-5
        },
        "image_token_index": 9,
        "boi_token_id": 10,
        "eoi_token_id": 11
        ,"audio_token_id": 12,
        "boa_token_id": 13,
        "eoa_token_id": 14,
        "audio_soft_tokens_per_image": 15
    }));
    assert_eq!(overrides.vision_hidden_size, 3072);
    assert_eq!(overrides.image_size, 384);
    assert_eq!(overrides.image_token_id, 9);
    assert_eq!(overrides.boi_token_id, 10);
    assert_eq!(overrides.eoi_token_id, 11);
    assert_eq!(overrides.audio_token_id, 12);
    assert_eq!(overrides.boa_token_id, 13);
    assert_eq!(overrides.eoa_token_id, 14);
    assert_eq!(overrides.audio_soft_tokens_per_clip, 15);
    assert!((overrides.vision_rms_eps - 1e-5).abs() < f32::EPSILON);
}

#[test]
fn gemma3n_language_model_prefix_prefers_quantized_prefix_when_present() {
    let mut quantized = WeightMap::new();
    quantized.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    assert_eq!(
        gemma3n_language_model_prefix(&quantized),
        "language_model.model"
    );

    let mut dense = WeightMap::new();
    dense.insert(
        "language_model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    assert_eq!(gemma3n_language_model_prefix(&dense), "language_model");
}

#[test]
fn gemma3n_needs_conv_transpose_checks_reference_conv_shape() {
    let mut raw_weights = WeightMap::new();
    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 16, 3, 3], dtype::FLOAT32),
    );
    assert!(gemma3n_needs_conv_transpose(&raw_weights));

    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 3, 3, 16], dtype::FLOAT32),
    );
    assert!(!gemma3n_needs_conv_transpose(&raw_weights));
}

#[test]
fn sanitize_gemma3n_weights_strips_model_prefix_and_transposes_conv_weights() {
    let mut raw_weights = WeightMap::new();
    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 16, 3, 3], dtype::FLOAT32),
    );
    raw_weights.insert(
        "model.language_model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    raw_weights.insert(
        "model.audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight".to_string(),
        mlxcel_core::ones(&[1536, 1, 5], dtype::FLOAT32),
    );

    let sanitized = sanitize_gemma3n_weights(raw_weights);

    assert!(sanitized.contains_key("vision_tower.timm_model.blocks.0.0.conv_exp.weight"));
    assert!(sanitized.contains_key("language_model.embed_tokens.weight"));
    assert_eq!(
        mlxcel_core::array_shape(
            sanitized
                .get("vision_tower.timm_model.blocks.0.0.conv_exp.weight")
                .unwrap()
        ),
        vec![8, 3, 3, 16]
    );
    assert_eq!(
        mlxcel_core::array_shape(
            sanitized
                .get("audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight")
                .unwrap()
        ),
        vec![1536, 5, 1]
    );
}

fn audio_conv_shape(weights: &WeightMap, key: &str) -> Vec<i32> {
    mlxcel_core::array_shape(weights.get(key).unwrap())
}

const LAYER0_CONV: &str = "audio_tower.subsample_conv_projection.layer0.conv.weight";
const LAYER1_CONV: &str = "audio_tower.subsample_conv_projection.layer1.conv.weight";
const LCONV1D: &str = "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight";

#[test]
fn sanitize_gemma4_audio_weights_preserves_channel_last_checkpoint() {
    // The mlx-community/gemma-4-e4b-it-qat-4bit checkpoint stores these conv
    // weights already in MLX channel-last layout. The sanitizer must leave them
    // untouched so the audio subsample conv receives a C_in=1 weight matching
    // the C_in=1 input (regression for issue #428).
    let mut weights = WeightMap::new();
    weights.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 3, 3, 1], dtype::FLOAT32),
    );
    weights.insert(
        LAYER1_CONV.to_string(),
        mlxcel_core::ones(&[32, 3, 3, 128], dtype::FLOAT32),
    );
    weights.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 5, 1], dtype::FLOAT32),
    );

    sanitize_gemma4_audio_weights(&mut weights);

    assert_eq!(audio_conv_shape(&weights, LAYER0_CONV), vec![128, 3, 3, 1]);
    assert_eq!(audio_conv_shape(&weights, LAYER1_CONV), vec![32, 3, 3, 128]);
    assert_eq!(audio_conv_shape(&weights, LCONV1D), vec![1024, 5, 1]);
}

#[test]
fn sanitize_gemma4_audio_weights_transposes_pytorch_layout() {
    // Synthetic PyTorch-layout weights: conv2d [out, in, kH, kW] and depthwise
    // conv1d [out, in=1, kW]. Both must be transposed to channel-last.
    let mut weights = WeightMap::new();
    weights.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 1, 3, 3], dtype::FLOAT32),
    );
    weights.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 1, 5], dtype::FLOAT32),
    );

    sanitize_gemma4_audio_weights(&mut weights);

    // conv2d [128, 1, 3, 3] --transpose[0,2,3,1]--> [128, 3, 3, 1].
    assert_eq!(audio_conv_shape(&weights, LAYER0_CONV), vec![128, 3, 3, 1]);
    // depthwise conv1d [1024, 1, 5] --transpose[0,2,1]--> [1024, 5, 1].
    assert_eq!(audio_conv_shape(&weights, LCONV1D), vec![1024, 5, 1]);
}

#[test]
fn sanitize_gemma4_audio_weights_is_idempotent() {
    // Running the sanitizer twice must equal running it once: the first pass
    // converts the PyTorch-layout weights to channel-last, and the second pass
    // must detect that layout and leave them unchanged.
    let mut once = WeightMap::new();
    once.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 1, 3, 3], dtype::FLOAT32),
    );
    once.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 1, 5], dtype::FLOAT32),
    );
    sanitize_gemma4_audio_weights(&mut once);

    let mut twice = WeightMap::new();
    twice.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 1, 3, 3], dtype::FLOAT32),
    );
    twice.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 1, 5], dtype::FLOAT32),
    );
    sanitize_gemma4_audio_weights(&mut twice);
    sanitize_gemma4_audio_weights(&mut twice);

    assert_eq!(
        audio_conv_shape(&once, LAYER0_CONV),
        audio_conv_shape(&twice, LAYER0_CONV)
    );
    assert_eq!(
        audio_conv_shape(&once, LCONV1D),
        audio_conv_shape(&twice, LCONV1D)
    );
    // And the idempotent result is the channel-last layout.
    assert_eq!(audio_conv_shape(&twice, LAYER0_CONV), vec![128, 3, 3, 1]);
    assert_eq!(audio_conv_shape(&twice, LCONV1D), vec![1024, 5, 1]);
}

/// Who wrote an NVFP4 export: the two producers store one weight format under
/// different tensor names and declare it differently.
#[derive(Debug, Clone, Copy)]
enum Nvfp4Producer {
    ModelOpt,
    CompressedTensors,
}

impl Nvfp4Producer {
    fn quantization_config(self) -> serde_json::Value {
        match self {
            Self::ModelOpt => json!({
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": { "group_0": { "weights": {
                    "num_bits": 4, "type": "float", "group_size": 16
                } } },
                "producer": { "name": "modelopt" }
            }),
            Self::CompressedTensors => json!({
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "quantization_status": "compressed",
                "config_groups": { "group_0": { "weights": {
                    "num_bits": 4, "type": "float", "group_size": 16
                } } }
            }),
        }
    }

    /// Insert one NVFP4 linear (8 x 64) under `base` with tensor scale 0.5,
    /// under this producer's tensor names.
    fn insert_linear(self, weights: &mut WeightMap, base: &str) {
        let codes: Vec<u8> = (0..8 * 32).map(|i| (i * 37 + 19) as u8).collect();
        let block_scales: Vec<u8> = (0..8 * 4).map(|i| [0x20, 0x28, 0x30][i % 3]).collect();
        let codes = mlxcel_core::from_bytes(&codes, &[8, 32], dtype::UINT8);
        let block_scales = mlxcel_core::from_bytes(&block_scales, &[8, 4], dtype::UINT8);
        weights.insert(format!("{base}.weight_scale"), block_scales);
        match self {
            Self::ModelOpt => {
                weights.insert(format!("{base}.weight"), codes);
                weights.insert(
                    format!("{base}.weight_scale_2"),
                    mlxcel_core::from_slice_f32(&[0.5], &[1]),
                );
            }
            Self::CompressedTensors => {
                weights.insert(format!("{base}.weight_packed"), codes);
                weights.insert(
                    format!("{base}.weight_global_scale"),
                    mlxcel_core::from_slice_f32(&[2.0], &[1]),
                );
                weights.insert(
                    format!("{base}.input_global_scale"),
                    mlxcel_core::from_slice_f32(&[37.5], &[1]),
                );
            }
        }
    }
}

/// The float tensors of [`nvfp4_gemma3_vlm_checkpoint`], under the names the
/// loader reads them by.
const GEMMA3_VLM_FLOAT_KEYS: [&str; 3] = [
    "model.norm.weight",
    "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
    "multi_modal_projector.mm_input_projection_weight",
];

/// A Gemma 3 VLM checkpoint in the `transformers` key layout with one NVFP4
/// linear in the language model, a float text tensor, a float vision linear
/// and the float projector, as `NeoChen1024/gemma-3-12b-it-NVFP4` has them.
fn nvfp4_gemma3_vlm_checkpoint(producer: Nvfp4Producer) -> WeightMap {
    let mut weights = WeightMap::new();
    producer.insert_linear(&mut weights, "language_model.model.layers.0.mlp.gate_proj");
    weights.insert(
        "language_model.model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 64], &[64]),
    );
    weights.insert(
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.25; 16], &[4, 4]),
    );
    weights.insert(
        "multi_modal_projector.mm_input_projection_weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.5; 8], &[2, 4]),
    );
    weights
}

fn gemma3_vlm_config_declaring(
    quantization_config: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut config = gemma3_vlm_config();
    if let Some(quantization_config) = quantization_config {
        config["quantization_config"] = quantization_config;
    }
    config
}

fn tensor_bytes(weights: &WeightMap, key: &str) -> Vec<u8> {
    let array = weights.get(key).unwrap_or_else(|| panic!("missing {key}"));
    mlxcel_core::eval(array);
    mlxcel_core::array_to_raw_bytes(array)
}

/// The Gemma 3 VLM route did not repack an NVFP4 export. A compressed-tensors
/// one (`NeoChen1024/gemma-3-12b-it-NVFP4`) failed with `Weight not found`; a
/// ModelOpt one would have reached the linear loaders as packed bytes. Both
/// are now repacked as on the text-only route, to the same planes, and the
/// loader is told so it can give the text config the layout.
#[test]
fn an_nvfp4_checkpoint_is_repacked_on_the_gemma3_vlm_route() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = "model.layers.0.mlp.gate_proj";
    let mut planes = Vec::new();
    for producer in [Nvfp4Producer::ModelOpt, Nvfp4Producer::CompressedTensors] {
        let checkpoint = nvfp4_gemma3_vlm_checkpoint(producer);
        let float_before: Vec<Vec<u8>> = GEMMA3_VLM_FLOAT_KEYS
            .iter()
            .map(|key| {
                let stored = if key.starts_with("model.") {
                    format!("language_model.{key}")
                } else {
                    key.to_string()
                };
                tensor_bytes(&checkpoint, &stored)
            })
            .collect();
        let full_config = gemma3_vlm_config_declaring(Some(producer.quantization_config()));
        let (weights, external_nvfp4) =
            prepare_gemma3_vlm_weights(checkpoint, &full_config, dir.path()).expect("prepare");
        assert!(external_nvfp4, "{producer:?}");
        for gone in [
            "weight_packed",
            "weight_global_scale",
            "input_global_scale",
            "weight_scale",
            "weight_scale_2",
        ] {
            assert!(
                !weights.contains_key(&format!("{base}.{gone}")),
                "{producer:?}: {gone}"
            );
        }
        assert!(
            weights.contains_key(&format!("{base}.scales")),
            "{producer:?}"
        );
        let weight = weights.get(&format!("{base}.weight")).expect("weight");
        assert_ne!(
            mlxcel_core::array_dtype(weight),
            dtype::UINT8,
            "{producer:?}: the packed bytes were left as they were"
        );
        // The float tensors are routed as before and keep their bytes.
        for (key, before) in GEMMA3_VLM_FLOAT_KEYS.iter().zip(&float_before) {
            assert_eq!(&tensor_bytes(&weights, key), before, "{producer:?}: {key}");
        }

        let mut keys: Vec<String> = weights.keys().cloned().collect();
        keys.sort();
        let bytes: Vec<Vec<u8>> = keys.iter().map(|key| tensor_bytes(&weights, key)).collect();
        planes.push((keys, bytes));
    }
    assert_eq!(
        planes[0], planes[1],
        "the two producers write one weight format"
    );

    // The text config of such an export has no MLX `quantization` block; the
    // loader gives it the repacked layout, and leaves a declared one alone.
    let text_config = |quantization: Option<serde_json::Value>| {
        let mut config = json!({ "model_type": "gemma3_text", "hidden_size": 64 });
        if let Some(quantization) = quantization {
            config["quantization"] = quantization;
        }
        serde_json::from_value::<crate::models::gemma3::ModelArgs>(config).expect("text config")
    };
    let mut undeclared = text_config(None);
    assert!(undeclared.quantization.is_none());
    assert_eq!((undeclared.group_size(), undeclared.bits()), (64, 4));
    undeclared.declare_modelopt_nvfp4_layout();
    assert_eq!((undeclared.group_size(), undeclared.bits()), (16, 4));

    let mut declared = text_config(Some(json!({ "group_size": 64, "bits": 8 })));
    declared.declare_modelopt_nvfp4_layout();
    assert_eq!((declared.group_size(), declared.bits()), (64, 8));
}

/// A checkpoint that does not declare NVFP4 keeps every tensor as the file
/// has it, whatever the tensors are named.
#[test]
fn the_gemma3_vlm_route_repacks_only_what_the_config_declares() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = "model.layers.0.mlp.gate_proj";
    for quantization_config in [
        None,
        Some(json!({ "quant_method": "modelopt", "quant_algo": "FP8" })),
        Some(json!({ "quant_method": "other" })),
        Some(json!({ "quant_method": "compressed-tensors", "format": "pack-quantized" })),
    ] {
        for (producer, kept) in [
            (Nvfp4Producer::ModelOpt, "weight_scale_2"),
            (Nvfp4Producer::CompressedTensors, "weight_packed"),
        ] {
            let (weights, external_nvfp4) = prepare_gemma3_vlm_weights(
                nvfp4_gemma3_vlm_checkpoint(producer),
                &gemma3_vlm_config_declaring(quantization_config.clone()),
                dir.path(),
            )
            .expect("prepare");
            assert!(!external_nvfp4, "{quantization_config:?}");
            assert!(
                weights.contains_key(&format!("{base}.{kept}")),
                "{quantization_config:?} {producer:?}"
            );
            assert!(
                !weights.contains_key(&format!("{base}.scales")),
                "{quantization_config:?} {producer:?}"
            );
            for key in GEMMA3_VLM_FLOAT_KEYS {
                assert!(weights.contains_key(key), "{quantization_config:?} {key}");
            }
        }
    }
}

/// ModelOpt also declares the format in an `hf_quant_config.json` next to
/// `config.json`; the route reads it as the text-only route does.
#[test]
fn the_gemma3_vlm_route_reads_the_modelopt_sidecar_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("hf_quant_config.json"),
        r#"{"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4", "group_size": 16}}"#,
    )
    .expect("sidecar");
    let (weights, external_nvfp4) = prepare_gemma3_vlm_weights(
        nvfp4_gemma3_vlm_checkpoint(Nvfp4Producer::ModelOpt),
        &gemma3_vlm_config_declaring(None),
        dir.path(),
    )
    .expect("prepare");
    assert!(external_nvfp4);
    assert!(weights.contains_key("model.layers.0.mlp.gate_proj.scales"));
}
