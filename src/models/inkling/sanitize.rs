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

use std::collections::BTreeMap;
use std::path::Path;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

type ExpertParts = BTreeMap<String, UniquePtr<MlxArray>>;

pub(crate) fn promote_nvfp4_config(
    path: &Path,
    config: &mut serde_json::Value,
) -> Result<(), String> {
    if config
        .get("quantization")
        .is_some_and(|value| !value.is_null())
        || config
            .get("quantization_config")
            .is_some_and(|value| !value.is_null())
    {
        return Ok(());
    }
    let hf_path = path.join("hf_quant_config.json");
    let raw = match std::fs::read_to_string(hf_path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("Failed to read hf_quant_config.json: {error}")),
    };
    let hf: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("Failed to parse hf_quant_config.json: {e}"))?;
    let quantization = hf.get("quantization").unwrap_or(&hf);
    let is_nvfp4 = quantization
        .get("quant_algo")
        .and_then(|v| v.as_str())
        .is_some_and(|v| v.eq_ignore_ascii_case("NVFP4"));
    if is_nvfp4 {
        let group_size = match quantization.get("group_size") {
            Some(value) => value.as_i64().ok_or_else(|| {
                "hf_quant_config.json NVFP4 group_size must be an integer".to_string()
            })?,
            None => 16,
        };
        if group_size != 16 {
            return Err(format!(
                "Inkling native NVFP4 requires group_size 16, but hf_quant_config.json declares {group_size}"
            ));
        }
        config
            .as_object_mut()
            .ok_or_else(|| "Inkling config must be an object".to_string())?
            .insert(
                "quantization".into(),
                serde_json::json!({"group_size": group_size, "bits": 4, "mode": "nvfp4"}),
            );
    }
    Ok(())
}

pub(crate) fn sanitize_weights(weights: WeightMap) -> Result<WeightMap, String> {
    let mut out = WeightMap::new();
    let mut experts: BTreeMap<usize, ExpertParts> = BTreeMap::new();
    for (key, value) in weights {
        if should_drop(&key) {
            continue;
        }
        if !key.starts_with("model.llm.") {
            out.insert(key, value);
            continue;
        }
        if key == "model.llm.embed.weight" {
            out.insert("model.embed_tokens.weight".into(), value);
            continue;
        }
        if key == "model.llm.unembed.weight" {
            out.insert("lm_head.weight".into(), value);
            continue;
        }
        if key == "model.llm.embed_norm.weight" {
            out.insert("model.embed_norm.weight".into(), value);
            continue;
        }
        if key == "model.llm.norm.weight" {
            out.insert("model.norm.weight".into(), value);
            continue;
        }
        let Some(rest) = key.strip_prefix("model.llm.layers.") else {
            continue;
        };
        let Some((index, sub)) = rest.split_once('.') else {
            continue;
        };
        let index: usize = index
            .parse()
            .map_err(|_| format!("Invalid Inkling layer key: {key}"))?;
        let prefix = format!("model.layers.{index}.");
        map_layer_weight(index, &prefix, sub, value, &mut out, &mut experts)?;
    }
    for (index, parts) in experts {
        emit_routed_experts(index, parts, &mut out)?;
    }
    Ok(out)
}

fn should_drop(key: &str) -> bool {
    key.starts_with("model.mtp.")
        || key.starts_with("model.visual.")
        || key.starts_with("model.audio.")
        || key.contains("training_args")
        || key.ends_with(".input_amax")
        || key.ends_with(".original_shape")
}

fn map_layer_weight(
    index: usize,
    prefix: &str,
    sub: &str,
    value: UniquePtr<MlxArray>,
    out: &mut WeightMap,
    experts: &mut BTreeMap<usize, ExpertParts>,
) -> Result<(), String> {
    if let Some(attention) = sub.strip_prefix("attn.") {
        let mapped = map_attention(attention);
        let value = if mapped.contains("sconv.conv.weight") {
            conv_to_mlx(&value)
        } else {
            value
        };
        out.insert(format!("{prefix}{mapped}"), value);
        return Ok(());
    }
    for (old, new) in [
        ("attn_norm.weight", "input_layernorm.weight"),
        ("mlp_norm.weight", "post_attention_layernorm.weight"),
        ("attn_sconv.weight", "attn_sconv.conv.weight"),
        ("mlp_sconv.weight", "mlp_sconv.conv.weight"),
    ] {
        if sub == old {
            let value = if old.contains("sconv") {
                conv_to_mlx(&value)
            } else {
                value
            };
            out.insert(format!("{prefix}{new}"), value);
            return Ok(());
        }
    }
    let Some(mlp) = sub.strip_prefix("mlp.") else {
        out.insert(format!("{prefix}{sub}"), value);
        return Ok(());
    };
    let mlp_prefix = format!("{prefix}mlp.");
    if mlp == "w13_dn.weight" {
        let (gate, up) = split_dense_w13(&value)?;
        out.insert(format!("{mlp_prefix}gate_proj.weight"), gate);
        out.insert(format!("{mlp_prefix}up_proj.weight"), up);
    } else if mlp == "w2_md.weight" {
        out.insert(format!("{mlp_prefix}down_proj.weight"), value);
    } else if mlp == "gate.weight" {
        out.insert(format!("{mlp_prefix}gate_weight"), value);
    } else if mlp == "gate.bias" {
        out.insert(
            format!("{mlp_prefix}e_score_correction_bias"),
            mlxcel_core::astype(&value, dtype::FLOAT32),
        );
    } else if matches!(mlp, "gate.global_scale" | "global_scale") {
        out.insert(format!("{mlp_prefix}global_scale"), value);
    } else if let Some(raw) = mlp.strip_prefix("experts.w13_weight") {
        experts
            .entry(index)
            .or_default()
            .insert(format!("w13{}", sidecar_leaf(raw)), value);
    } else if let Some(raw) = mlp.strip_prefix("experts.w2_weight") {
        experts
            .entry(index)
            .or_default()
            .insert(format!("w2{}", sidecar_leaf(raw)), value);
    } else if let Some(rest) = mlp.strip_prefix("experts.") {
        out.insert(format!("{mlp_prefix}switch_mlp.{rest}"), value);
    } else if mlp == "shared_experts.shared_w13_weight" {
        let (gate, up) = split_expert_w13(&value)?;
        out.insert(
            format!("{mlp_prefix}shared_experts.gate_proj.weight"),
            merge_expert_rows(&gate)?,
        );
        out.insert(
            format!("{mlp_prefix}shared_experts.up_proj.weight"),
            merge_expert_rows(&up)?,
        );
    } else if mlp == "shared_experts.shared_w2_weight" {
        out.insert(
            format!("{mlp_prefix}shared_experts.down_proj.weight"),
            merge_expert_down(&value)?,
        );
    } else if let Some(rest) = mlp.strip_prefix("shared_experts.") {
        let merged = if rest.starts_with("down_proj.") {
            merge_expert_down(&value)?
        } else {
            merge_expert_rows(&value)?
        };
        out.insert(format!("{mlp_prefix}shared_experts.{rest}"), merged);
    } else {
        out.insert(format!("{mlp_prefix}{mlp}"), value);
    }
    Ok(())
}

fn map_attention(key: &str) -> String {
    if key == "rel_logits_proj.proj" {
        return "self_attn.rel_proj".into();
    }
    for (old, new) in [
        ("wq_du", "q_proj"),
        ("wk_dv", "k_proj"),
        ("wv_dv", "v_proj"),
        ("wr_du", "r_proj"),
        ("wo_ud", "o_proj"),
        ("q_norm", "q_norm"),
        ("k_norm", "k_norm"),
        ("k_sconv", "k_sconv.conv"),
        ("v_sconv", "v_sconv.conv"),
    ] {
        if key == old {
            return format!("self_attn.{new}");
        }
        if let Some(rest) = key.strip_prefix(&format!("{old}.")) {
            return format!("self_attn.{new}.{rest}");
        }
    }
    format!("self_attn.{key}")
}

fn conv_to_mlx(value: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(value);
    if shape.len() == 3 && shape[2] != 1 {
        mlxcel_core::transpose_axes(value, &[0, 2, 1])
    } else {
        mlxcel_core::copy(value)
    }
}

fn sidecar_leaf(rest: &str) -> &str {
    if rest.is_empty() { ".weight" } else { rest }
}

fn split_dense_w13(value: &MlxArray) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
    let shape = mlxcel_core::array_shape(value);
    if shape.len() != 2 || shape[0] % 2 != 0 {
        return Err(format!("Inkling dense w13 must be [2I,H], got {shape:?}"));
    }
    let paired = mlxcel_core::reshape(value, &[shape[0] / 2, 2, shape[1]]);
    let gate = mlxcel_core::squeeze_axis(&mlxcel_core::utils::slice_axis(&paired, 1, 0, 1), 1);
    let up = mlxcel_core::squeeze_axis(&mlxcel_core::utils::slice_axis(&paired, 1, 1, 2), 1);
    Ok((
        mlxcel_core::contiguous(&gate, false),
        mlxcel_core::contiguous(&up, false),
    ))
}

fn split_expert_w13(
    value: &MlxArray,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
    let shape = mlxcel_core::array_shape(value);
    if shape.len() != 3 || shape[1] % 2 != 0 {
        return Err(format!(
            "Inkling expert w13 must be [E,2I,H], got {shape:?}"
        ));
    }
    let paired = mlxcel_core::reshape(value, &[shape[0], shape[1] / 2, 2, shape[2]]);
    let gate = mlxcel_core::squeeze_axis(&mlxcel_core::utils::slice_axis(&paired, 2, 0, 1), 2);
    let up = mlxcel_core::squeeze_axis(&mlxcel_core::utils::slice_axis(&paired, 2, 1, 2), 2);
    Ok((
        mlxcel_core::contiguous(&gate, false),
        mlxcel_core::contiguous(&up, false),
    ))
}

fn merge_expert_rows(value: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(value);
    if shape.len() != 3 {
        return Ok(mlxcel_core::copy(value));
    }
    Ok(mlxcel_core::reshape(
        value,
        &[shape[0] * shape[1], shape[2]],
    ))
}

fn merge_expert_down(value: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(value);
    if shape.len() != 3 {
        return Ok(mlxcel_core::copy(value));
    }
    let transposed = mlxcel_core::transpose_axes(value, &[1, 0, 2]);
    Ok(mlxcel_core::reshape(
        &transposed,
        &[shape[1], shape[0] * shape[2]],
    ))
}

fn emit_routed_experts(
    index: usize,
    mut parts: ExpertParts,
    out: &mut WeightMap,
) -> Result<(), String> {
    let prefix = format!("model.layers.{index}.mlp.switch_mlp");
    let w13 = parts
        .remove("w13.weight")
        .ok_or_else(|| format!("layer {index}: missing routed w13 weight"))?;
    let w2 = parts
        .remove("w2.weight")
        .ok_or_else(|| format!("layer {index}: missing routed w2 weight"))?;
    let w13_shape = mlxcel_core::array_shape(&w13);
    let w2_shape = mlxcel_core::array_shape(&w2);
    validate_routed_weight_shapes(index, &w13_shape, &w2_shape)?;
    let experts = w13_shape[0];
    if (mlxcel_core::array_dtype(&w13) == dtype::UINT8)
        != (mlxcel_core::array_dtype(&w2) == dtype::UINT8)
    {
        return Err(format!(
            "layer {index}: routed w13 and w2 weights cannot mix native NVFP4 bytes with regular tensors"
        ));
    }
    if mlxcel_core::array_dtype(&w13) == dtype::UINT8 {
        if w13_shape[2] % 4 != 0 || w2_shape[2] % 4 != 0 {
            return Err(format!(
                "layer {index}: NVFP4 packed weight dimensions must be divisible by 4, got w13 {w13_shape:?} and w2 {w2_shape:?}"
            ));
        }
        let raw_scale13 = parts
            .remove("w13.scale")
            .ok_or_else(|| format!("layer {index}: missing routed w13 scale"))?;
        let raw_scale2 = parts
            .remove("w2.scale")
            .ok_or_else(|| format!("layer {index}: missing routed w2 scale"))?;
        let raw_scale2_13 = parts
            .remove("w13.scale2")
            .ok_or_else(|| format!("layer {index}: missing routed w13 scale2"))?;
        let raw_scale2_2 = parts
            .remove("w2.scale2")
            .ok_or_else(|| format!("layer {index}: missing routed w2 scale2"))?;
        if !parts.is_empty() {
            return Err(format!(
                "layer {index}: unexpected native NVFP4 routed sidecars: {:?}",
                parts.keys().collect::<Vec<_>>()
            ));
        }
        validate_nvfp4_sidecar_shapes(
            index,
            &w13_shape,
            &w2_shape,
            &raw_scale13,
            &raw_scale2,
            &raw_scale2_13,
            &raw_scale2_2,
        )?;
        let w13 = mlxcel_core::view(&w13, dtype::UINT32);
        let w2 = mlxcel_core::view(&w2, dtype::UINT32);
        let (gate, up) = split_expert_w13(&w13)?;
        out.insert(format!("{prefix}.gate_proj.weight"), gate);
        out.insert(format!("{prefix}.up_proj.weight"), up);
        out.insert(format!("{prefix}.down_proj.weight"), w2);
        let scale13 = normalized_e4m3(raw_scale13)?;
        let (gate_scale, up_scale) = split_expert_w13(&scale13)?;
        out.insert(format!("{prefix}.gate_proj.scales"), gate_scale);
        out.insert(format!("{prefix}.up_proj.scales"), up_scale);
        out.insert(
            format!("{prefix}.down_proj.scales"),
            normalized_e4m3(raw_scale2)?,
        );
        let scale2_13 = mlxcel_core::astype(&raw_scale2_13, dtype::FLOAT32);
        let scale2_2 = mlxcel_core::astype(&raw_scale2_2, dtype::FLOAT32);
        out.insert(
            format!("{prefix}.gate_scale"),
            mlxcel_core::copy(&scale2_13),
        );
        out.insert(
            format!("{prefix}.out_scale"),
            mlxcel_core::multiply(&scale2_13, &scale2_2),
        );
    } else {
        if !parts.is_empty() {
            return Err(format!(
                "layer {index}: regular routed experts cannot carry native NVFP4 sidecars: {:?}",
                parts.keys().collect::<Vec<_>>()
            ));
        }
        let (gate, up) = split_expert_w13(&w13)?;
        out.insert(format!("{prefix}.gate_proj.weight"), gate);
        out.insert(format!("{prefix}.up_proj.weight"), up);
        out.insert(format!("{prefix}.down_proj.weight"), w2);
        out.insert(
            format!("{prefix}.gate_scale"),
            mlxcel_core::ones(&[experts], dtype::FLOAT32),
        );
        out.insert(
            format!("{prefix}.out_scale"),
            mlxcel_core::ones(&[experts], dtype::FLOAT32),
        );
    }
    Ok(())
}

fn validate_routed_weight_shapes(index: usize, w13: &[i32], w2: &[i32]) -> Result<(), String> {
    if w13.len() != 3 || w13[0] <= 0 || w13[1] <= 0 || w13[1] % 2 != 0 || w13[2] <= 0 {
        return Err(format!(
            "layer {index}: routed w13 must be [E,2I,H], got {w13:?}"
        ));
    }
    if w2.len() != 3 || w2[0] != w13[0] || w2[1] <= 0 || w2[2] <= 0 {
        return Err(format!(
            "layer {index}: routed w2 must be [E,H,I] with the same expert count as w13, got {w2:?}"
        ));
    }
    Ok(())
}

fn validate_nvfp4_sidecar_shapes(
    index: usize,
    w13: &[i32],
    w2: &[i32],
    scale13: &MlxArray,
    scale2: &MlxArray,
    scale2_13: &MlxArray,
    scale2_2: &MlxArray,
) -> Result<(), String> {
    let scale13_shape = mlxcel_core::array_shape(scale13);
    let scale2_shape = mlxcel_core::array_shape(scale2);
    let scale2_13_shape = mlxcel_core::array_shape(scale2_13);
    let scale2_2_shape = mlxcel_core::array_shape(scale2_2);
    if scale13_shape.len() != 3
        || scale13_shape[0] != w13[0]
        || scale13_shape[1] != w13[1]
        || scale13_shape[2] <= 0
    {
        return Err(format!(
            "layer {index}: routed w13 scale must be [E,2I,G], got {scale13_shape:?}"
        ));
    }
    if scale2_shape.len() != 3
        || scale2_shape[0] != w2[0]
        || scale2_shape[1] != w2[1]
        || scale2_shape[2] <= 0
    {
        return Err(format!(
            "layer {index}: routed w2 scale must be [E,H,G], got {scale2_shape:?}"
        ));
    }
    let expected = [w13[0]];
    if scale2_13_shape != expected || scale2_2_shape != expected {
        return Err(format!(
            "layer {index}: routed scale2 sidecars must be [E], got w13 {scale2_13_shape:?} and w2 {scale2_2_shape:?}"
        ));
    }
    Ok(())
}

fn normalized_e4m3(value: UniquePtr<MlxArray>) -> Result<UniquePtr<MlxArray>, String> {
    if mlxcel_core::array_dtype(&value) == dtype::UINT8 {
        return Ok(value);
    }
    let shape = mlxcel_core::array_shape(&value);
    let f32_value = mlxcel_core::astype(&value, dtype::FLOAT32);
    let raw = mlxcel_core::array_to_raw_bytes(&f32_value);
    let mut encoded = Vec::with_capacity(raw.len() / 4);
    for bytes in raw.chunks_exact(4) {
        encoded.push(crate::models::sanitize::f32_to_f8_e4m3(f32::from_ne_bytes(
            bytes.try_into().unwrap(),
        )));
    }
    Ok(mlxcel_core::from_bytes(&encoded, &shape, dtype::UINT8))
}
