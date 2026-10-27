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

use serde_json::json;

use super::*;

fn quant_json(format: &str, bits: i64, group_size: i64) -> Value {
    json!({
        "format": format,
        "solver": "ternary",
        "bits": bits,
        "group_size": group_size,
        "scale_dtype": "bfloat16",
        "skip_patterns": ["proj_out", "x_embedder"],
        "quantized_modules": ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_out.0"],
    })
}

#[test]
fn reads_bits_group_size_and_the_module_list() {
    let c = PackedQuantConfig::from_json(&quant_json(PACKED_FORMAT, 2, 128)).unwrap();
    assert_eq!(
        c.quant,
        QuantParams {
            bits: 2,
            group_size: 128
        }
    );
    assert_eq!(c.modules.len(), 2);
    assert!(c.modules.contains("transformer_blocks.0.attn.to_out.0"));
}

#[test]
fn rejects_another_format_naming_it() {
    let err = PackedQuantConfig::from_json(&quant_json("gemlite-ternary", 2, 128)).unwrap_err();
    assert!(err.to_string().contains("gemlite-ternary"), "{err}");
}

#[test]
fn rejects_unsupported_bits_and_group_sizes_naming_the_value() {
    // The binary 1-bit variant: mlxcel-core's affine path has no 1-bit kernel.
    let err = PackedQuantConfig::from_json(&quant_json(PACKED_FORMAT, 1, 128)).unwrap_err();
    assert!(err.to_string().contains("bits (1)"), "{err}");
    let err = PackedQuantConfig::from_json(&quant_json(PACKED_FORMAT, 2, 256)).unwrap_err();
    assert!(err.to_string().contains("group_size (256)"), "{err}");
}

#[test]
fn a_missing_module_list_is_an_error() {
    let mut v = quant_json(PACKED_FORMAT, 2, 128);
    v.as_object_mut().unwrap().remove("quantized_modules");
    let err = PackedQuantConfig::from_json(&v).unwrap_err();
    assert!(err.to_string().contains("quantized_modules"), "{err}");
}

#[test]
fn layout_names_the_first_missing_component() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join(TRANSFORMER_DIR)).unwrap();
    let err = layout(root).unwrap_err().to_string();
    assert!(err.contains(TEXT_ENCODER_DIR), "{err}");
    for dir in [TEXT_ENCODER_DIR, "vae", "tokenizer"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    let err = layout(root).unwrap_err().to_string();
    assert!(err.contains(QUANT_CONFIG_FILE), "{err}");
    std::fs::write(root.join(TRANSFORMER_DIR).join(QUANT_CONFIG_FILE), "{}").unwrap();
    std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
    let l = layout(root).unwrap();
    assert_eq!(l.transformer, root.join(TRANSFORMER_DIR));
    assert_eq!(l.text_encoder, root.join(TEXT_ENCODER_DIR));
}
