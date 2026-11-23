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

//! Inkling 2 quantization metadata.
//!
//! The native release declares `quantization_config: {modelopt, NVFP4}` in
//! `config.json`, but that single block is wrong for Mini: its experts are
//! NVFP4 in layers 0-19 and MXFP8 in layers 20-23. The authoritative table is
//! `hf_quant_config.json` (`quant_method: inkling2`), one parameter group per
//! weight transform. This mirrors mlx-vlm's `Model.quantization_from_metadata`
//! (`mlx_vlm/models/inkling2/inkling2.py`, MIT).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

/// Quantization of one stacked expert projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertQuant {
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'static str,
}

pub const NVFP4: ExpertQuant = ExpertQuant {
    group_size: 16,
    bits: 4,
    mode: "nvfp4",
};
pub const MXFP8: ExpertQuant = ExpertQuant {
    group_size: 32,
    bits: 8,
    mode: "mxfp8",
};

/// Per-layer expert quantization from `hf_quant_config.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inkling2QuantTable {
    /// `(layer, projection)` -> settings, projection one of
    /// `gate_proj`, `up_proj`, `down_proj`.
    entries: BTreeMap<(usize, &'static str), ExpertQuant>,
}

impl Inkling2QuantTable {
    pub fn get(&self, layer: usize, projection: &str) -> Option<ExpertQuant> {
        self.entries
            .iter()
            .find(|((l, p), _)| *l == layer && *p == projection)
            .map(|(_, quant)| *quant)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Read `hf_quant_config.json` when the checkpoint ships one with
/// `quant_method: inkling2`. A missing file, or another quant method, is
/// `Ok(None)`; a malformed table or an unknown `weight_transform` is an error
/// naming it.
pub fn read_native_quant_table(model_path: &Path) -> Result<Option<Inkling2QuantTable>, String> {
    let path = model_path.join("hf_quant_config.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Failed to read {}: {error}", path.display())),
    };
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
    parse_native_quant_table(&value)
}

pub fn parse_native_quant_table(value: &Value) -> Result<Option<Inkling2QuantTable>, String> {
    if value.get("quant_method").and_then(Value::as_str) != Some("inkling2") {
        return Ok(None);
    }
    let groups = value
        .get("model")
        .and_then(|model| model.get("parameter_groups"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "hf_quant_config.json: model.parameter_groups must be an array".to_string()
        })?;
    let mut table = Inkling2QuantTable::default();
    for (index, group) in groups.iter().enumerate() {
        let kind = group
            .get("weight_transform")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!("hf_quant_config.json: parameter group {index} has no weight_transform")
            })?;
        let settings = match kind {
            // Safetensors stores these groups in their declared dtype; the
            // loader keeps it (see `bf16_to_f16_at_load`).
            "cast" => continue,
            "nvfp4_nearest" => NVFP4,
            "mxfp8_nearest" => MXFP8,
            other => {
                return Err(format!(
                    "Unsupported Inkling 2 weight transform {other:?} in hf_quant_config.json \
                     parameter group {index}"
                ));
            }
        };
        let tensors = group
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!("hf_quant_config.json: parameter group {index} has no tensors")
            })?;
        for tensor in tensors {
            let name = tensor.as_str().ok_or_else(|| {
                format!("hf_quant_config.json: group {index} lists a non-string tensor")
            })?;
            let (layer, projections) = expert_target(name).ok_or_else(|| {
                format!("hf_quant_config.json: {kind} tensor {name:?} is not a routed expert plane")
            })?;
            for projection in projections {
                crate::models::switch_layers::validate_expert_quantization_params(
                    name,
                    settings.group_size,
                    settings.bits,
                )?;
                table.entries.insert((layer, projection), settings);
            }
        }
    }
    Ok(Some(table))
}

/// `model.llm.layers.N.mlp.experts.w13_weight` -> `(N, [gate_proj, up_proj])`.
fn expert_target(name: &str) -> Option<(usize, &'static [&'static str])> {
    let rest = name
        .strip_prefix("model.llm.layers.")
        .or_else(|| name.strip_prefix("model.mtp.layers."))?;
    let (layer, leaf) = rest.split_once('.')?;
    let layer = layer.parse().ok()?;
    match leaf {
        "mlp.experts.w13_weight" => Some((layer, &["gate_proj", "up_proj"])),
        "mlp.experts.w2_weight" => Some((layer, &["down_proj"])),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_like() -> Value {
        let nvfp4: Vec<String> = (0..20)
            .flat_map(|i| {
                [
                    format!("model.llm.layers.{i}.mlp.experts.w13_weight"),
                    format!("model.llm.layers.{i}.mlp.experts.w2_weight"),
                ]
            })
            .collect();
        let mxfp8: Vec<String> = (20..24)
            .flat_map(|i| {
                [
                    format!("model.llm.layers.{i}.mlp.experts.w13_weight"),
                    format!("model.llm.layers.{i}.mlp.experts.w2_weight"),
                ]
            })
            .collect();
        serde_json::json!({
            "quant_method": "inkling2",
            "model": {"parameter_groups": [
                {"weight_transform": "cast", "tensors": ["model.llm.attn_res.output_query"]},
                {"weight_transform": "nvfp4_nearest", "tensors": nvfp4},
                {"weight_transform": "mxfp8_nearest", "tensors": mxfp8},
            ]}
        })
    }

    #[test]
    fn mini_table_is_nvfp4_then_mxfp8_per_projection() {
        let table = parse_native_quant_table(&mini_like()).unwrap().unwrap();
        assert_eq!(table.len(), 24 * 3);
        for layer in 0..20 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                assert_eq!(
                    table.get(layer, projection),
                    Some(NVFP4),
                    "{layer} {projection}"
                );
            }
        }
        for layer in 20..24 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                assert_eq!(
                    table.get(layer, projection),
                    Some(MXFP8),
                    "{layer} {projection}"
                );
            }
        }
    }

    #[test]
    fn unknown_transform_is_an_error_naming_it() {
        let value = serde_json::json!({
            "quant_method": "inkling2",
            "model": {"parameter_groups": [
                {"weight_transform": "int4_awq", "tensors": ["model.llm.layers.0.mlp.experts.w2_weight"]}
            ]}
        });
        let error = parse_native_quant_table(&value).unwrap_err();
        assert!(error.contains("int4_awq"), "{error}");
    }

    #[test]
    fn other_quant_methods_are_not_this_table() {
        let value = serde_json::json!({"quant_method": "modelopt", "quantization": {}});
        assert_eq!(parse_native_quant_table(&value).unwrap(), None);
    }
}
