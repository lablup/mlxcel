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

use super::*;

fn dir_named(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join(name);
    std::fs::create_dir_all(&path).unwrap();
    (tmp, path)
}

#[test]
fn variant_follows_the_directory_name() {
    for (name, want) in [
        ("FLUX.2-klein-4B-bf16", Flux2Variant::Klein4b),
        ("flux2-klein-4b-4bit", Flux2Variant::Klein4b),
        ("flux2-klein-9b-4bit", Flux2Variant::Klein9b),
        ("FLUX.2-klein-base-4B", Flux2Variant::KleinBase4b),
        ("FLUX.2-klein-base-9B", Flux2Variant::KleinBase9b),
        ("FLUX.2-klein-9b-kv", Flux2Variant::Klein9bKv),
    ] {
        let (_tmp, path) = dir_named(name);
        assert_eq!(Flux2Variant::from_dir(&path).unwrap(), want, "{name}");
    }
}

#[test]
fn kv_weights_file_selects_the_kv_variant() {
    let (_tmp, path) = dir_named("klein");
    std::fs::write(path.join("flux-2-klein-9b-kv.safetensors"), b"").unwrap();
    assert_eq!(
        Flux2Variant::from_dir(&path).unwrap(),
        Flux2Variant::Klein9bKv
    );
}

#[test]
fn unnamed_dir_uses_is_distilled_and_the_transformer_shape() {
    let (_tmp, path) = dir_named("checkpoint");
    assert!(Flux2Variant::from_dir(&path).is_err());
    std::fs::create_dir_all(path.join("transformer")).unwrap();
    std::fs::write(
        path.join("transformer/config.json"),
        r#"{"num_layers": 8, "num_attention_heads": 32}"#,
    )
    .unwrap();
    assert_eq!(
        Flux2Variant::from_dir(&path).unwrap(),
        Flux2Variant::Klein9b
    );
    std::fs::write(path.join("model_index.json"), r#"{"is_distilled": false}"#).unwrap();
    assert_eq!(
        Flux2Variant::from_dir(&path).unwrap(),
        Flux2Variant::KleinBase9b
    );
}

#[test]
fn unnamed_dir_falls_back_to_the_text_encoder_width() {
    let (_tmp, path) = dir_named("checkpoint");
    std::fs::create_dir_all(path.join("text_encoder")).unwrap();
    std::fs::write(
        path.join("text_encoder/config.json"),
        r#"{"hidden_size": 2560}"#,
    )
    .unwrap();
    assert_eq!(
        Flux2Variant::from_dir(&path).unwrap(),
        Flux2Variant::Klein4b
    );
}

#[test]
fn defaults_per_variant() {
    assert_eq!(Flux2Variant::Klein4b.default_steps(), 4);
    assert_eq!(Flux2Variant::Klein4b.default_guidance(), 1.0);
    assert_eq!(Flux2Variant::KleinBase4b.default_steps(), 50);
    assert_eq!(Flux2Variant::KleinBase4b.default_guidance(), 4.0);
    assert!(Flux2Variant::Klein9bKv.uses_reference_kv_cache());
    assert!(!Flux2Variant::Klein9b.uses_reference_kv_cache());
    let nine = Flux2Variant::Klein9bKv.transformer_config();
    assert_eq!((nine.num_layers, nine.num_single_layers), (8, 24));
    assert_eq!(nine.inner_dim(), 4096);
    let four = Flux2Variant::Klein4b.transformer_config();
    assert_eq!((four.inner_dim(), four.mlp_hidden_dim()), (3072, 9216));
}

#[test]
fn transformer_config_reads_diffusers_json_or_falls_back() {
    let tmp = tempfile::tempdir().unwrap();
    let config = Flux2TransformerConfig::resolve(tmp.path(), Flux2Variant::Klein9b).unwrap();
    assert_eq!(config, Flux2Variant::Klein9b.transformer_config());
    std::fs::write(
        tmp.path().join("config.json"),
        r#"{"_class_name": "Flux2Transformer2DModel", "num_layers": 5, "num_single_layers": 20,
            "num_attention_heads": 24, "joint_attention_dim": 7680, "out_channels": null,
            "axes_dims_rope": [32, 32, 32, 32], "rope_theta": 2000, "guidance_embeds": false}"#,
    )
    .unwrap();
    let config = Flux2TransformerConfig::resolve(tmp.path(), Flux2Variant::Klein9b).unwrap();
    assert_eq!(config, Flux2Variant::Klein4b.transformer_config());
    assert_eq!(config.out_channels(), 128);
    std::fs::write(
        tmp.path().join("config.json"),
        r#"{"axes_dims_rope": [32, 32]}"#,
    )
    .unwrap();
    assert!(Flux2TransformerConfig::resolve(tmp.path(), Flux2Variant::Klein4b).is_err());
}

#[test]
fn dimensions_name_the_bounds() {
    assert!(validate_dimensions(512, 512).is_ok());
    assert!(validate_dimensions(256, 2048).is_ok());
    let err = validate_dimensions(240, 512).unwrap_err().to_string();
    assert!(err.contains("[256, 2048]"), "{err}");
    let err = validate_dimensions(512, 4096).unwrap_err().to_string();
    assert!(err.contains("height must be in [256, 2048]"), "{err}");
    let err = validate_dimensions(520, 512).unwrap_err().to_string();
    assert!(err.contains("multiple of 16"), "{err}");
}

#[test]
fn layout_names_the_missing_component() {
    let tmp = tempfile::tempdir().unwrap();
    for dir in ["transformer", "text_encoder", "vae", "tokenizer"] {
        let err = Flux2Layout::from_dir(tmp.path()).unwrap_err().to_string();
        assert!(err.contains(dir), "{err}");
        std::fs::create_dir_all(tmp.path().join(dir)).unwrap();
    }
    let err = Flux2Layout::from_dir(tmp.path()).unwrap_err().to_string();
    assert!(err.contains("tokenizer.json"), "{err}");
    std::fs::write(tmp.path().join("tokenizer/tokenizer.json"), "{}").unwrap();
    assert!(Flux2Layout::from_dir(tmp.path()).is_ok());
}

#[test]
fn quantization_metadata_is_read_like_upstream() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(read_quant_params(tmp.path()).unwrap(), None);
    let index = tmp.path().join("model.safetensors.index.json");
    std::fs::write(
        &index,
        r#"{"metadata": {"quantization_level": "4", "quantization_group_size": "64"}, "weight_map": {}}"#,
    )
    .unwrap();
    assert_eq!(
        read_quant_params(tmp.path()).unwrap(),
        Some(QuantParams {
            bits: 4,
            group_size: 64
        })
    );
    std::fs::write(
        &index,
        r#"{"metadata": {"quantization": {"bits": 8, "group_size": 32, "mode": "affine"}}}"#,
    )
    .unwrap();
    assert_eq!(
        read_quant_params(tmp.path()).unwrap(),
        Some(QuantParams {
            bits: 8,
            group_size: 32
        })
    );
    std::fs::write(&index, r#"{"metadata": {"bits": 4, "mode": "mxfp4"}}"#).unwrap();
    assert!(read_quant_params(tmp.path()).is_err());
}

#[test]
fn text_encoder_config_is_synthesized_for_quantized_exports() {
    let tmp = tempfile::tempdir().unwrap();
    let q = Some(QuantParams {
        bits: 4,
        group_size: 64,
    });
    let config = text_encoder_config(tmp.path(), Flux2Variant::Klein9b, q).unwrap();
    assert_eq!(config["hidden_size"], 4096);
    assert_eq!(config["intermediate_size"], 12288);
    assert_eq!(config["num_hidden_layers"], 36);
    assert_eq!(config["quantization"]["group_size"], 64);
    let args: crate::models::qwen3::ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.head_dim, 128);
}
