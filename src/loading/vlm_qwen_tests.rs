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

//! Qwen-VL family loader tests.
//!
//! The synthetic half asserts the key remap that lets `mlxcel generate` and
//! `mlxcel-server` read a raw `transformers` export of Qwen2.5-VL. The real
//! half is one `#[ignore]`d differential gate that runs the generation path
//! on both a raw export and its mlx conversion and requires the same tokens
//! out of each; it soft-skips when either checkpoint is absent.

use super::{
    prepare_qwen2_5_vl_weights, prepare_qwen3_vl_weights, qwen_vision_config,
    remap_qwen2_5_vl_native_keys, rewrite_qwen2_5_vl_native_key, without_duplicate_in_chans,
};

#[test]
fn native_key_remap_renames_the_tower_and_unwraps_the_language_model() {
    // Older exports: `visual.*` beside a bare `model.*` decoder. Newer ones:
    // `model.visual.*` beside `model.language_model.*`. Both reach this path
    // because `strip_language_model_prefix` only handles a leading
    // `language_model.`.
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("visual.blocks.0.attn.qkv.weight"),
        "vision_tower.blocks.0.attn.qkv.weight"
    );
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("model.visual.patch_embed.proj.weight"),
        "vision_tower.patch_embed.proj.weight"
    );
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("model.language_model.layers.0.mlp.up_proj.weight"),
        "model.layers.0.mlp.up_proj.weight"
    );
    assert_eq!(
        rewrite_qwen2_5_vl_native_key("lm_head.weight"),
        "lm_head.weight"
    );
}

#[test]
fn native_key_remap_maps_a_whole_export_and_leaves_a_conversion_alone() {
    let dummy = || mlxcel_core::from_slice_f32(&[0.0], &[1]);
    let mut raw = mlxcel_core::weights::WeightMap::new();
    for key in [
        "visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "lm_head.weight",
    ] {
        raw.insert(key.to_string(), dummy());
    }

    let remapped = remap_qwen2_5_vl_native_keys(raw);
    let mut keys: Vec<&str> = remapped.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "lm_head.weight",
            "model.layers.0.mlp.up_proj.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
            "vision_tower.patch_embed.proj.weight",
        ]
    );

    // An mlx-community conversion already carries the target names, the
    // merger included, so every rule misses and the map comes back unchanged.
    let mut converted = mlxcel_core::weights::WeightMap::new();
    for key in [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.merger.mlp.0.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.embed_tokens.weight",
    ] {
        converted.insert(key.to_string(), dummy());
    }
    let before: Vec<String> = {
        let mut keys: Vec<String> = converted.keys().cloned().collect();
        keys.sort_unstable();
        keys
    };
    let after = {
        let mut keys: Vec<String> = remap_qwen2_5_vl_native_keys(converted)
            .keys()
            .cloned()
            .collect();
        keys.sort_unstable();
        keys
    };
    assert_eq!(before, after);
}

// Real-checkpoint gate. `#[ignore]`d because it loads two 3B checkpoints, and
// it soft-skips (rather than failing) when either directory is absent.

/// Repo ids the gate looks for through the usual store lookup.
const RAW_EXPORT: &str = "Qwen/Qwen2.5-VL-3B-Instruct";
const MLX_CONVERSION: &str = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16";

/// Environment overrides for a checkout that keeps these two snapshots
/// somewhere the store lookup does not reach, for example a shared
/// `models/mlx/...` tree. Each names a model directory.
const RAW_EXPORT_DIR_ENV: &str = "MLXCEL_TEST_QWEN25VL_RAW_DIR";
const MLX_CONVERSION_DIR_ENV: &str = "MLXCEL_TEST_QWEN25VL_MLX_DIR";

/// The prompt both checkpoints get, already in Qwen2.5-VL chat-template form
/// so the gate does not depend on the tokenizer's template rendering.
const IMAGE_PROMPT: &str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n";

/// A deterministic image. The gate asserts two checkpoints agree on the same
/// input, so the picture only has to be identical between the two runs.
fn gate_image() -> image::DynamicImage {
    let mut buffer = image::RgbImage::new(112, 112);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x * 2) as u8, (y * 2) as u8, ((x + y) % 256) as u8]);
    }
    image::DynamicImage::ImageRgb8(buffer)
}

fn gate_checkpoint(repo_id: &str, env_key: &str) -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var(env_key) {
        let dir = std::path::PathBuf::from(dir);
        if dir.join("config.json").is_file() {
            return Some(dir);
        }
        eprintln!(
            "skipping real-checkpoint gate: {env_key}={} has no config.json",
            dir.display()
        );
        return None;
    }
    crate::models::embedding_test_support::local_checkpoint(repo_id)
}

/// Greedy tokens for one checkpoint, through the same calls `mlxcel generate
/// --image` makes: `load_model`, the shared VLM embedding preparation, and a
/// backend session seeded with those embeddings.
fn greedy_image_tokens(model_dir: &std::path::Path, max_tokens: usize) -> Vec<i32> {
    let (model, tokenizer) =
        crate::loading::load_model(model_dir).expect("the Qwen2.5-VL checkpoint loads");
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode_with_special(IMAGE_PROMPT, false, true)
        .expect("the chat-template prompt tokenizes")
        .iter()
        .map(|&id| id as i32)
        .collect();

    let prepared = crate::vlm_runtime::prepare_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        IMAGE_PROMPT,
        &[gate_image()],
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&id| id as i32)
                .collect()
        },
    )
    .expect("the image request prepares")
    .expect("a Qwen2.5-VL image request must produce vision embeddings");

    let (input_embeds, mask) = crate::vlm_runtime::prepared_embedding_refs(&prepared.embeddings)
        .expect("prepared embeddings expose an input tensor");
    let mut session = crate::backend::select_backend()
        .create_session(
            model_dir,
            mlxcel_core::generate::LanguageModel::num_layers(&model),
            mlxcel_core::cache::KVCacheMode::Fp16,
            mlxcel_core::sampling::TokenBiasMap::new(),
        )
        .expect("a generation session is created");

    session.generate_streaming_with_embeddings(
        &model,
        &prompt_tokens,
        Some(input_embeds),
        mask,
        max_tokens,
        &mlxcel_core::generate::SamplingConfig::greedy(),
        |_| true,
    )
}

/// A raw `transformers` export and its mlx conversion are the same weights
/// under different names and a different `Conv3d` filter layout, so the
/// generation path must produce the same greedy tokens from both. Before the
/// remap and the layout normalization landed, the raw export did not load at
/// all, and a hand-renamed one generated from scrambled filters.
#[test]
#[ignore = "loads two 3B Qwen2.5-VL checkpoints"]
fn qwen2_5_vl_raw_export_matches_mlx_conversion() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let (Some(raw_dir), Some(mlx_dir)) = (
        gate_checkpoint(RAW_EXPORT, RAW_EXPORT_DIR_ENV),
        gate_checkpoint(MLX_CONVERSION, MLX_CONVERSION_DIR_ENV),
    ) else {
        return;
    };

    let raw_tokens = greedy_image_tokens(&raw_dir, 24);
    let mlx_tokens = greedy_image_tokens(&mlx_dir, 24);
    assert!(
        !raw_tokens.is_empty(),
        "the raw export generated no tokens at all"
    );
    assert_eq!(
        raw_tokens, mlx_tokens,
        "the raw export and its mlx conversion must decode to the same tokens"
    );
}

/// A `transformers` re-export writes the patch-embedding channel count under
/// both `in_channels` and its older name `in_chans` (NVIDIA's ModelOpt
/// Qwen2.5-VL export does). The vision config takes either through a serde
/// alias, and serde refused the pair as a duplicate field, so the checkpoint
/// did not load.
#[test]
fn a_vision_config_naming_the_channel_count_twice_parses() {
    use crate::vision::encoders::qwen2_5_vl::Qwen25VLVisionConfig;
    use serde_json::json;

    let both = json!({ "vision_config": {
        "hidden_size": 1280, "in_channels": 3, "in_chans": 3
    } });
    // The bug: the struct itself still refuses the pair.
    assert!(serde_json::from_value::<Qwen25VLVisionConfig>(both["vision_config"].clone()).is_err());
    for has_vision in [true, false] {
        let config: Qwen25VLVisionConfig =
            qwen_vision_config(&both, has_vision, "vision config").expect("both names");
        assert_eq!(config.in_channels, 3, "has_vision {has_vision}");
    }
    // The IREE host-preprocessor loaders parse through the same dedupe.
    let deduplicated = without_duplicate_in_chans(&both);
    assert!(matches!(deduplicated, std::borrow::Cow::Owned(_)));
    assert!(deduplicated["vision_config"].get("in_chans").is_none());
    assert!(
        serde_json::from_value::<Qwen25VLVisionConfig>(deduplicated["vision_config"].clone())
            .is_ok()
    );

    // Either name alone is read as before.
    for (name, channels) in [("in_chans", 1), ("in_channels", 4)] {
        let mut full = json!({ "vision_config": { "hidden_size": 1280 } });
        full["vision_config"][name] = json!(channels);
        let config: Qwen25VLVisionConfig =
            qwen_vision_config(&full, true, "vision config").expect("one name");
        assert_eq!(config.in_channels, channels, "{name}");
        assert!(matches!(
            without_duplicate_in_chans(&full),
            std::borrow::Cow::Borrowed(_)
        ));
    }
}

/// A Qwen2.5-VL checkpoint as ModelOpt writes it: one NVFP4 linear (8 x 64,
/// two 4-bit codes per byte, one E4M3 scale per 16 elements, one scale for the
/// tensor) under the text model's keys, a float text tensor and a float vision
/// tensor under the native `visual.` prefix.
fn modelopt_qwen2_5_vl_checkpoint() -> mlxcel_core::weights::WeightMap {
    use mlxcel_core::dtype;

    let mut weights = mlxcel_core::weights::WeightMap::new();
    let base = "model.layers.0.mlp.gate_proj";
    let codes: Vec<u8> = (0..8 * 32).map(|i| (i * 37 + 19) as u8).collect();
    let block_scales: Vec<u8> = (0..8 * 4).map(|i| [0x20, 0x28, 0x30][i % 3]).collect();
    weights.insert(
        format!("{base}.weight"),
        mlxcel_core::from_bytes(&codes, &[8, 32], dtype::UINT8),
    );
    weights.insert(
        format!("{base}.weight_scale"),
        mlxcel_core::from_bytes(&block_scales, &[8, 4], dtype::UINT8),
    );
    weights.insert(
        format!("{base}.weight_scale_2"),
        mlxcel_core::from_slice_f32(&[0.5], &[1]),
    );
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 64], &[64]),
    );
    weights.insert(
        "visual.blocks.0.attn.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.25; 16], &[4, 4]),
    );
    weights
}

fn qwen2_5_vl_config(quantization_config: Option<serde_json::Value>) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "qwen2_5_vl",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "vocab_size": 16,
        "vision_config": { "hidden_size": 4 }
    });
    if let Some(quantization_config) = quantization_config {
        config["quantization_config"] = quantization_config;
    }
    config
}

fn modelopt_quantization_config(quant_algo: &str) -> serde_json::Value {
    serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": quant_algo,
        "config_groups": { "group_0": { "weights": {
            "num_bits": 4, "type": "float", "group_size": 16
        } } },
        "producer": { "name": "modelopt" }
    })
}

/// The Qwen2.5-VL route did not repack a ModelOpt NVFP4 export, so the packed
/// `uint8` codes reached the linear loaders as they were. It now repacks the
/// triplet as the text-only route does, and reports the format so the loader
/// can give the text config its layout.
#[test]
fn a_modelopt_nvfp4_checkpoint_is_repacked_on_the_qwen2_5_vl_route() {
    use mlxcel_core::dtype;

    let dir = tempfile::tempdir().expect("tempdir");
    let full_config = qwen2_5_vl_config(Some(modelopt_quantization_config("NVFP4")));
    let (weights, modelopt_nvfp4) =
        prepare_qwen2_5_vl_weights(modelopt_qwen2_5_vl_checkpoint(), &full_config, dir.path())
            .expect("prepare");
    assert!(modelopt_nvfp4);

    let base = "model.layers.0.mlp.gate_proj";
    for gone in ["weight_scale", "weight_scale_2"] {
        assert!(!weights.contains_key(&format!("{base}.{gone}")), "{gone}");
    }
    assert!(weights.contains_key(&format!("{base}.scales")), "scales");
    let weight = weights.get(&format!("{base}.weight")).expect("weight");
    assert_ne!(
        mlxcel_core::array_dtype(weight),
        dtype::UINT8,
        "the packed bytes were left as they were"
    );
    // Everything else is routed as before.
    assert!(weights.contains_key("model.norm.weight"));
    assert!(weights.contains_key("vision_tower.blocks.0.attn.proj.weight"));

    // The text config of such an export has no MLX `quantization` block; the
    // loader gives it the repacked layout, and leaves a declared one alone.
    let mut text_config: crate::models::qwen2_vl::Qwen2VLConfig =
        serde_json::from_value(full_config).expect("text config");
    assert!(text_config.quantization.is_none());
    text_config.declare_modelopt_nvfp4_layout();
    let layout = text_config.quantization.as_ref().expect("layout");
    assert_eq!((layout.group_size, layout.bits), (16, 4));

    let mut declared: crate::models::qwen2_vl::Qwen2VLConfig = serde_json::from_value({
        let mut config = qwen2_5_vl_config(None);
        config["quantization"] = serde_json::json!({ "group_size": 64, "bits": 8 });
        config
    })
    .expect("text config");
    declared.declare_modelopt_nvfp4_layout();
    let layout = declared.quantization.as_ref().expect("layout");
    assert_eq!((layout.group_size, layout.bits), (64, 8));
}

#[test]
fn the_qwen2_5_vl_route_repacks_only_what_the_config_declares() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = "model.layers.0.mlp.gate_proj";
    for quantization_config in [
        None,
        Some(modelopt_quantization_config("FP8")),
        Some(serde_json::json!({ "quant_method": "other" })),
    ] {
        let (weights, modelopt_nvfp4) = prepare_qwen2_5_vl_weights(
            modelopt_qwen2_5_vl_checkpoint(),
            &qwen2_5_vl_config(quantization_config.clone()),
            dir.path(),
        )
        .expect("prepare");
        assert!(!modelopt_nvfp4, "{quantization_config:?}");
        assert!(
            weights.contains_key(&format!("{base}.weight_scale_2")),
            "{quantization_config:?}"
        );
        assert!(
            !weights.contains_key(&format!("{base}.scales")),
            "{quantization_config:?}"
        );
    }
}

/// ModelOpt also writes the declaration to `hf_quant_config.json`; a config
/// without a `quantization_config` is recognized through that file.
#[test]
fn the_qwen2_5_vl_route_reads_the_modelopt_sidecar_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("hf_quant_config.json"),
        serde_json::to_vec(&serde_json::json!({
            "producer": { "name": "modelopt" },
            "quantization": { "quant_algo": "NVFP4", "group_size": 16 }
        }))
        .unwrap(),
    )
    .unwrap();
    let (weights, modelopt_nvfp4) = prepare_qwen2_5_vl_weights(
        modelopt_qwen2_5_vl_checkpoint(),
        &qwen2_5_vl_config(None),
        dir.path(),
    )
    .expect("prepare");
    assert!(modelopt_nvfp4);
    assert!(weights.contains_key("model.layers.0.mlp.gate_proj.scales"));
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
            Self::ModelOpt => modelopt_quantization_config("NVFP4"),
            Self::CompressedTensors => serde_json::json!({
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
    fn insert_linear(self, weights: &mut mlxcel_core::weights::WeightMap, base: &str) {
        use mlxcel_core::dtype;

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

/// A dense Qwen3-VL checkpoint in the `transformers` key layout with one NVFP4
/// linear in the language model, a float text tensor and a float vision
/// tensor.
fn nvfp4_qwen3_vl_checkpoint(producer: Nvfp4Producer) -> mlxcel_core::weights::WeightMap {
    let mut weights = mlxcel_core::weights::WeightMap::new();
    producer.insert_linear(&mut weights, "model.language_model.layers.0.mlp.gate_proj");
    weights.insert(
        "model.language_model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 64], &[64]),
    );
    weights.insert(
        "model.visual.blocks.0.attn.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.25; 16], &[4, 4]),
    );
    weights
}

fn qwen3_vl_config(quantization_config: Option<serde_json::Value>) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "vocab_size": 16
        },
        "vision_config": { "hidden_size": 4 }
    });
    if let Some(quantization_config) = quantization_config {
        config["quantization_config"] = quantization_config;
    }
    config
}

/// The dense Qwen3-VL route did not repack an NVFP4 export. A
/// compressed-tensors one (`JEILDLWLRMA/Qwen3-VL-4B-Instruct-NVFP4`) failed
/// with `Weight not found`; a ModelOpt one would have reached the linear
/// loaders as packed bytes. Both are now repacked as on the text-only route,
/// to the same planes, and the loader is told so it can give the text config
/// the layout.
#[test]
fn an_nvfp4_checkpoint_is_repacked_on_the_qwen3_vl_route() {
    use mlxcel_core::dtype;

    let dir = tempfile::tempdir().expect("tempdir");
    let base = "model.layers.0.mlp.gate_proj";
    let mut planes = Vec::new();
    for producer in [Nvfp4Producer::ModelOpt, Nvfp4Producer::CompressedTensors] {
        let full_config = qwen3_vl_config(Some(producer.quantization_config()));
        let (weights, external_nvfp4) = prepare_qwen3_vl_weights(
            nvfp4_qwen3_vl_checkpoint(producer),
            &full_config,
            dir.path(),
        )
        .expect("prepare");
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
        // Everything else is routed as before.
        assert!(weights.contains_key("model.norm.weight"), "{producer:?}");
        assert!(
            weights.contains_key("vision_tower.blocks.0.attn.proj.weight"),
            "{producer:?}"
        );

        let mut keys: Vec<&String> = weights.keys().collect();
        keys.sort();
        let bytes: Vec<Vec<u8>> = keys
            .iter()
            .map(|key| {
                let array = weights.get(key.as_str()).unwrap();
                mlxcel_core::eval(array);
                mlxcel_core::array_to_raw_bytes(array)
            })
            .collect();
        planes.push((keys.into_iter().cloned().collect::<Vec<_>>(), bytes));
    }
    assert_eq!(
        planes[0], planes[1],
        "the two producers write one weight format"
    );

    // The text config of such an export has no MLX `quantization` block; the
    // loader gives it the repacked layout, and leaves a declared one alone.
    let text_config = |quantization: Option<serde_json::Value>| {
        let mut config = qwen3_vl_config(None)["text_config"].clone();
        if let Some(quantization) = quantization {
            config["quantization"] = quantization;
        }
        serde_json::from_value::<crate::models::qwen3_vl::Qwen3VLConfig>(config)
            .expect("text config")
    };
    let mut undeclared = text_config(None);
    assert!(undeclared.quantization.is_none());
    undeclared.declare_modelopt_nvfp4_layout();
    let layout = undeclared.quantization.as_ref().expect("layout");
    assert_eq!((layout.group_size, layout.bits), (16, 4));

    let mut declared = text_config(Some(serde_json::json!({ "group_size": 64, "bits": 8 })));
    declared.declare_modelopt_nvfp4_layout();
    let layout = declared.quantization.as_ref().expect("layout");
    assert_eq!((layout.group_size, layout.bits), (64, 8));
}

#[test]
fn the_qwen3_vl_route_repacks_only_what_the_config_declares() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = "model.layers.0.mlp.gate_proj";
    for quantization_config in [
        None,
        Some(modelopt_quantization_config("FP8")),
        Some(serde_json::json!({ "quant_method": "other" })),
        Some(serde_json::json!({
            "quant_method": "compressed-tensors", "format": "pack-quantized"
        })),
    ] {
        for (producer, kept) in [
            (Nvfp4Producer::ModelOpt, "weight_scale_2"),
            (Nvfp4Producer::CompressedTensors, "weight_packed"),
        ] {
            let (weights, external_nvfp4) = prepare_qwen3_vl_weights(
                nvfp4_qwen3_vl_checkpoint(producer),
                &qwen3_vl_config(quantization_config.clone()),
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
        }
    }
}

/// The vision tower loads with the MLX `quantization` block's layout, which
/// an NVFP4 export does not have. One that also packs vision linears
/// (`Forturne/Qwen3-VL-Embedding-2B-NVFP4` does) is refused by name instead of
/// handing the tower repacked planes it would read as dense weights.
#[test]
fn an_nvfp4_export_with_a_packed_vision_tower_is_refused_on_the_qwen3_vl_route() {
    let dir = tempfile::tempdir().expect("tempdir");
    for producer in [Nvfp4Producer::ModelOpt, Nvfp4Producer::CompressedTensors] {
        let mut checkpoint = nvfp4_qwen3_vl_checkpoint(producer);
        producer.insert_linear(&mut checkpoint, "model.visual.blocks.0.attn.qkv");
        let error = match prepare_qwen3_vl_weights(
            checkpoint,
            &qwen3_vl_config(Some(producer.quantization_config())),
            dir.path(),
        ) {
            Ok(_) => panic!("{producer:?}: a packed vision tower must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("vision_tower.blocks.0.attn.qkv") && error.contains("vision tower"),
            "{producer:?}: {error}"
        );
    }
}
