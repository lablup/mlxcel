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

//! The Qwen3.5 VLM route repacks a ModelOpt NVFP4 export's text weights.

use std::path::Path;

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::qwen::split_qwen3_5_vlm_weights;

const LINEAR: &str = "layers.0.mlp.gate_proj";

/// A Qwen3.5 VLM checkpoint as ModelOpt writes it: one NVFP4 linear (8 x 64,
/// two 4-bit codes per byte, one E4M3 scale per 16 elements, one scale for the
/// tensor) under `model.language_model.`, a float text tensor, a float vision
/// tensor, and a tensor of the prediction head this loader does not use.
fn checkpoint() -> WeightMap {
    let mut weights = WeightMap::new();
    let base = format!("model.language_model.{LINEAR}");
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
        "model.language_model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 64], &[64]),
    );
    weights.insert(
        "model.visual.blocks.0.attn.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.25; 16], &[4, 4]),
    );
    weights.insert(
        "mtp.fc.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.25; 16], &[4, 4]),
    );
    weights
}

fn config(quantization_config: Option<serde_json::Value>) -> serde_json::Value {
    let mut config = json!({
        "model_type": "qwen3_5",
        "text_config": { "model_type": "qwen3_5_text" },
        "vision_config": {}
    });
    if let Some(quantization_config) = quantization_config {
        config["quantization_config"] = quantization_config;
    }
    config
}

fn modelopt(quant_algo: &str) -> serde_json::Value {
    json!({
        "quant_method": "modelopt",
        "quant_algo": quant_algo,
        "config_groups": { "group_0": { "weights": {
            "num_bits": 4, "type": "float", "group_size": 16
        } } },
        "producer": { "name": "modelopt" }
    })
}

/// A directory with no `hf_quant_config.json`, so only the config decides.
fn no_sidecar_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

#[test]
fn a_modelopt_nvfp4_checkpoint_is_repacked_on_the_vlm_route() {
    let dir = no_sidecar_dir();
    let (text, vision) =
        split_qwen3_5_vlm_weights(checkpoint(), &config(Some(modelopt("NVFP4"))), dir.path())
            .expect("split");

    // The triplet is gone and the linear is a quantized plane with scales,
    // under the text model's own key names.
    let base = format!("model.{LINEAR}");
    for gone in ["weight_scale", "weight_scale_2"] {
        assert!(!text.contains_key(&format!("{base}.{gone}")), "{gone}");
    }
    assert!(text.contains_key(&format!("{base}.scales")), "scales");
    let weight = text.get(&format!("{base}.weight")).expect("weight");
    assert_ne!(
        mlxcel_core::array_dtype(weight),
        dtype::UINT8,
        "the packed bytes were left as they were"
    );

    // Everything else is routed as before.
    assert!(text.contains_key("model.norm.weight"));
    assert!(vision.contains_key("vision_tower.blocks.0.attn.proj.weight"));
    assert!(
        !text
            .keys()
            .chain(vision.keys())
            .any(|key| key.starts_with("mtp."))
    );
}

#[test]
fn the_vlm_route_repacks_only_what_the_config_declares() {
    let dir = no_sidecar_dir();
    let base = format!("model.{LINEAR}");
    for quantization_config in [
        None,
        Some(modelopt("FP8")),
        Some(json!({ "quant_method": "other" })),
    ] {
        let (text, _vision) = split_qwen3_5_vlm_weights(
            checkpoint(),
            &config(quantization_config.clone()),
            dir.path(),
        )
        .expect("split");
        assert!(
            text.contains_key(&format!("{base}.weight_scale_2")),
            "{quantization_config:?}"
        );
        assert!(
            !text.contains_key(&format!("{base}.scales")),
            "{quantization_config:?}"
        );
    }
}

/// ModelOpt also writes the declaration to `hf_quant_config.json`; a config
/// without a `quantization_config` is recognized through that file, as on the
/// text-only route.
#[test]
fn the_vlm_route_reads_the_modelopt_sidecar_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        Path::new(dir.path()).join("hf_quant_config.json"),
        serde_json::to_vec(&json!({
            "producer": { "name": "modelopt" },
            "quantization": { "quant_algo": "NVFP4", "group_size": 16 }
        }))
        .unwrap(),
    )
    .unwrap();
    let (text, _vision) =
        split_qwen3_5_vlm_weights(checkpoint(), &config(None), dir.path()).expect("split");
    assert!(text.contains_key(&format!("model.{LINEAR}.scales")));
}
