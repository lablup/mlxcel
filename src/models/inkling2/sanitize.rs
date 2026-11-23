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

//! Inkling 2 native-checkpoint key remap.
//!
//! Maps both published layouts onto the names an mlx-vlm conversion stores
//! (`language_model.*`, `vision_tower.*`), so every checkpoint loads through
//! one path:
//!
//! - the native release (`model.llm.*`, `model.visual.*`, fused interleaved
//!   `w13` expert bytes with `_amax / _outer_scale_denominator`), mapped as
//!   mlx-vlm `Model.sanitize` does (`mlx_vlm/models/inkling2/inkling2.py`, MIT);
//! - the Transformers-layout MLX conversion (`model.language_model.*`,
//!   `model.vision_tower.*`, top-level `lm_head.*`, `*_message_gate`,
//!   `post_attention_convolution`, packed `gate_up_proj` with an MLX
//!   `global_scale`), which mlx-vlm 466069f5 does not read. Its expert bytes,
//!   scales and message queries are bit-identical to the native release's
//!   (checked against `models/inkling-2-mini`), and its `global_scale` is
//!   exactly `2688 * amax / denominator`.
//!
//! Keys already in the converted namespace pass through unchanged.

use std::collections::BTreeMap;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use crate::models::inkling::sanitize::{conv_to_mlx, normalized_e4m3, split_expert_w13};

pub(crate) const TEXT_PREFIX: &str = "language_model.model.";
pub(crate) const HEAD_KEY: &str = "language_model.lm_head.weight";

/// MLX's NVFP4 kernels dequantize with `global_scale / (448 * 6)`.
const NVFP4_GLOBAL_SCALE_FACTOR: f32 = 2688.0;

type ExpertParts = BTreeMap<String, UniquePtr<MlxArray>>;

pub(crate) fn sanitize_weights(
    weights: WeightMap,
    hidden_size: usize,
) -> Result<WeightMap, String> {
    let mut out = WeightMap::new();
    let mut experts: BTreeMap<usize, ExpertParts> = BTreeMap::new();
    let mut cross_pass: BTreeMap<&'static str, UniquePtr<MlxArray>> = BTreeMap::new();
    for (key, value) in weights {
        if key.starts_with("model.mtp.") || key.starts_with("mtp.") {
            continue;
        }
        if let Some(rest) = key.strip_prefix("model.llm.layers.") {
            let (layer, sub) = rest
                .split_once('.')
                .ok_or_else(|| format!("Invalid Inkling 2 layer key: {key}"))?;
            let index: usize = layer
                .parse()
                .map_err(|_| format!("Invalid Inkling 2 layer key: {key}"))?;
            if let Some(leaf) = sub.strip_prefix("mlp.experts.") {
                experts
                    .entry(index)
                    .or_default()
                    .insert(leaf.to_string(), value);
                continue;
            }
            let (target, value) = map_layer_key(&key, sub, value)?;
            insert(
                &mut out,
                format!("{TEXT_PREFIX}layers.{index}.{target}"),
                value,
            )?;
        } else if let Some(sub) = key.strip_prefix("model.llm.attn_res.") {
            let target = map_attn_res_key(&key, sub)?;
            insert(&mut out, format!("{TEXT_PREFIX}{target}"), value)?;
        } else if let Some(sub) = key.strip_prefix("model.llm.") {
            match sub {
                "unembed.weight" => insert(&mut out, HEAD_KEY.into(), value)?,
                "embed.weight" => {
                    insert(&mut out, format!("{TEXT_PREFIX}embed_tokens.weight"), value)?
                }
                "embed_norm.weight" | "norm.weight" => {
                    insert(&mut out, format!("{TEXT_PREFIX}{sub}"), value)?
                }
                "middle_cycle_cross_pass_queries" => {
                    cross_pass.insert("attn_mlp", value);
                }
                "middle_cycle_cross_pass_kv_queries" => {
                    cross_pass.insert("kv", value);
                }
                _ => return Err(format!("Unmapped Inkling 2 checkpoint key: {key}")),
            }
        } else if let Some(sub) = key.strip_prefix("model.visual.") {
            insert(
                &mut out,
                format!("vision_tower.{}", map_visual_key(sub)),
                value,
            )?;
        } else if let Some(sub) = key.strip_prefix("model.language_model.") {
            map_hf_text_key(sub, value, &mut out)?;
        } else if let Some(sub) = key.strip_prefix("model.vision_tower.") {
            let sub = sub
                .replace("merger.linear_1.", "proj_in.")
                .replace("merger.linear_2.", "proj_out.")
                .replace(
                    "embeddings.position_embedding.weight",
                    "embeddings.position_embedding",
                );
            insert(&mut out, format!("vision_tower.{sub}"), value)?;
        } else if let Some(leaf) = key.strip_prefix("lm_head.") {
            insert(&mut out, format!("language_model.lm_head.{leaf}"), value)?;
        } else {
            // mlx-vlm converted keys (and anything outside the native
            // namespaces) pass through unchanged, which makes the remap a
            // no-op on a converted checkpoint.
            insert(&mut out, key, value)?;
        }
    }
    for (index, parts) in experts {
        emit_experts(index, parts, &mut out)?;
    }
    fuse_message_queries(cross_pass, hidden_size, &mut out)?;
    Ok(out)
}

fn insert(out: &mut WeightMap, key: String, value: UniquePtr<MlxArray>) -> Result<(), String> {
    if out.insert(key.clone(), value).is_some() {
        return Err(format!("Inkling 2 checkpoint maps two tensors onto {key}"));
    }
    Ok(())
}

fn map_layer_key(
    key: &str,
    sub: &str,
    value: UniquePtr<MlxArray>,
) -> Result<(String, UniquePtr<MlxArray>), String> {
    if let Some(attn) = sub.strip_prefix("attn.") {
        let mapped = match attn {
            "wq_du.weight" => "self_attn.q_proj.weight",
            "wk_dv.weight" => "self_attn.k_proj.weight",
            "wv_dv.weight" => "self_attn.v_proj.weight",
            "wr_du.weight" => "self_attn.r_proj.weight",
            "wo_ud.weight" => "self_attn.o_proj.weight",
            "gate_attn.weight" => "self_attn.gate_proj.weight",
            "attn_sink_proj.weight" => "self_attn.sink_proj.weight",
            "q_l2_clip.scale" => "self_attn.q_scale",
            "rel_l2_clip.scale" => "self_attn.rel_scale",
            "rel_logits_proj.proj" => "self_attn.rel_proj",
            "k_sconv.weight" => {
                return Ok(("self_attn.k_sconv.conv.weight".into(), conv_to_mlx(&value)));
            }
            "v_sconv.weight" => {
                return Ok(("self_attn.v_sconv.conv.weight".into(), conv_to_mlx(&value)));
            }
            "k_sconv.static_w" => "self_attn.k_sconv.static_weight",
            "v_sconv.static_w" => "self_attn.v_sconv.static_weight",
            "k_sconv.weight_proj" => "self_attn.k_sconv.weight_proj.weight",
            "v_sconv.weight_proj" => "self_attn.v_sconv.weight_proj.weight",
            _ => return Err(format!("Unmapped Inkling 2 checkpoint key: {key}")),
        };
        return Ok((mapped.into(), value));
    }
    let mapped = match sub {
        "attn_sconv.weight" => return Ok(("attn_sconv.conv.weight".into(), conv_to_mlx(&value))),
        "mlp_sconv.weight" => return Ok(("mlp_sconv.conv.weight".into(), conv_to_mlx(&value))),
        "mlp.gate.weight" => "mlp.gate_weight",
        "mlp.gate.bias" => "mlp.e_score_correction_bias",
        "attn_norm.weight" => "input_layernorm.weight",
        "mlp_norm.weight" => "post_attention_layernorm.weight",
        "q_input_norm.weight" => "q_input_layernorm.weight",
        "k_input_norm.weight" => "k_input_layernorm.weight",
        "v_input_norm.weight" => "v_input_layernorm.weight",
        other => {
            let controller = other.strip_prefix("controllers.").and_then(|rest| {
                let (name, leaf) = rest.split_once('.')?;
                (matches!(name, "attn" | "k" | "v" | "mlp")
                    && matches!(leaf, "down_weight" | "forget_weight" | "gain_weight"))
                .then_some(())
            });
            if controller.is_some() {
                return Ok((other.to_string(), value));
            }
            return Err(format!("Unmapped Inkling 2 checkpoint key: {key}"));
        }
    };
    Ok((mapped.into(), value))
}

/// One `model.language_model.*` tensor of the Transformers-layout conversion.
fn map_hf_text_key(
    sub: &str,
    value: UniquePtr<MlxArray>,
    out: &mut WeightMap,
) -> Result<(), String> {
    if sub == "message_input.query" {
        return insert(out, format!("{TEXT_PREFIX}message_queries"), value);
    }
    let Some(rest) = sub.strip_prefix("layers.") else {
        return insert(out, format!("{TEXT_PREFIX}{sub}"), value);
    };
    let (layer, leaf) = rest
        .split_once('.')
        .ok_or_else(|| format!("Invalid Inkling 2 layer key: model.language_model.{sub}"))?;
    let base = format!("{TEXT_PREFIX}layers.{layer}");
    if let Some(tensor) = leaf.strip_prefix("mlp.experts.gate_up_proj.") {
        let switch = format!("{base}.mlp.switch_mlp");
        if tensor == "global_scale" {
            insert(
                out,
                format!("{switch}.gate_proj.native_global_scale"),
                mlxcel_core::copy(&value),
            )?;
            return insert(out, format!("{switch}.up_proj.native_global_scale"), value);
        }
        // Packed codes and scales keep the native interleaved row order.
        let (gate, up) = split_expert_w13(&value)?;
        insert(out, format!("{switch}.gate_proj.{tensor}"), gate)?;
        return insert(out, format!("{switch}.up_proj.{tensor}"), up);
    }
    if let Some(tensor) = leaf.strip_prefix("mlp.experts.down_proj.") {
        let tensor = if tensor == "global_scale" {
            "native_global_scale"
        } else {
            tensor
        };
        return insert(
            out,
            format!("{base}.mlp.switch_mlp.down_proj.{tensor}"),
            value,
        );
    }
    let mapped = match leaf {
        "mlp.gate.weight" => "mlp.gate_weight".to_string(),
        "mlp.gate.e_score_correction_bias" => "mlp.e_score_correction_bias".to_string(),
        "post_attention_convolution.conv1d.weight" => "attn_sconv.conv.weight".to_string(),
        "post_mlp_convolution.conv1d.weight" => "mlp_sconv.conv.weight".to_string(),
        "self_attn.k_sconv.conv1d.weight" => "self_attn.k_sconv.conv.weight".to_string(),
        "self_attn.v_sconv.conv1d.weight" => "self_attn.v_sconv.conv.weight".to_string(),
        "self_attn.q_l2_clip.scale" => "self_attn.q_scale".to_string(),
        "self_attn.rel_l2_clip.scale" => "self_attn.rel_scale".to_string(),
        "self_attn.rel_logits_proj.proj" => "self_attn.rel_proj".to_string(),
        other => match other.split_once("_message_gate.") {
            Some((gate, proj)) => {
                let controller = match gate {
                    "q" => "attn",
                    "k" | "v" | "mlp" => gate,
                    _ => {
                        return Err(format!(
                            "Unmapped Inkling 2 checkpoint key: model.language_model.{sub}"
                        ));
                    }
                };
                let weight = match proj {
                    "down_proj.weight" => "down_weight",
                    "forget_gate_proj.weight" => "forget_weight",
                    "message_gate_proj.weight" => "gain_weight",
                    _ => {
                        return Err(format!(
                            "Unmapped Inkling 2 checkpoint key: model.language_model.{sub}"
                        ));
                    }
                };
                format!("controllers.{controller}.{weight}")
            }
            None => other.to_string(),
        },
    };
    insert(out, format!("{base}.{mapped}"), value)
}

/// `attn_queries.I` is layer `I + 1`'s attention input (layer 0 has none),
/// `mlp_queries.I` layer `I`'s MLP input, `output_query` the final input.
fn map_attn_res_key(key: &str, sub: &str) -> Result<String, String> {
    if sub == "output_query" {
        return Ok("final_input.query".into());
    }
    let (kind, index) = sub
        .split_once('.')
        .ok_or_else(|| format!("Unmapped Inkling 2 checkpoint key: {key}"))?;
    let index: usize = index
        .parse()
        .map_err(|_| format!("Unmapped Inkling 2 checkpoint key: {key}"))?;
    match kind {
        "attn_queries" => Ok(format!("layers.{}.attention_input.query", index + 1)),
        "mlp_queries" => Ok(format!("layers.{index}.mlp_input.query")),
        _ => Err(format!("Unmapped Inkling 2 checkpoint key: {key}")),
    }
}

fn map_visual_key(sub: &str) -> String {
    let sub = sub.strip_prefix("backbone.").unwrap_or(sub);
    let sub = sub
        .replace("ln_pre.", "pre_layernorm.")
        .replace(".ln_1.", ".layernorm_before.")
        .replace(".ln_2.", ".layernorm_after.");
    sub.replace(".ls_1.gamma", ".lambda_1")
        .replace(".ls_2.gamma", ".lambda_2")
        .replace(".mlp.c_fc.", ".mlp.fc1.")
        .replace(".mlp.c_proj.", ".mlp.fc2.")
}

/// Split `w13` into gate and up and reinterpret packed planes.
///
/// A plane with a `_scale` sidecar stores its packed codes as bytes; MLX's
/// quantized kernels read them as `uint32`. `_amax / _outer_scale_denominator`
/// is the NVFP4 per-expert tensor scale (`weight_scale_2`), shared by gate and
/// up; it is handed to `gather_qmm` as MLX's native `global_scale`
/// (`2688 * weight_scale_2`), exactly as mlx-vlm's `ScaledQuantizedSwitchLinear`.
fn emit_experts(index: usize, mut parts: ExpertParts, out: &mut WeightMap) -> Result<(), String> {
    let base = format!("{TEXT_PREFIX}layers.{index}.mlp.switch_mlp");
    for (name, projections) in [
        ("w13_weight", &["gate_proj", "up_proj"][..]),
        ("w2_weight", &["down_proj"][..]),
    ] {
        let weight = parts
            .remove(name)
            .ok_or_else(|| format!("Inkling 2 layer {index}: missing mlp.experts.{name}"))?;
        let scale = parts.remove(&format!("{name}_scale"));
        let amax = parts.remove(&format!("{name}_amax"));
        let denominator = parts.remove(&format!("{name}_outer_scale_denominator"));
        let weight_shape = mlxcel_core::array_shape(&weight);
        if weight_shape.len() != 3 || weight_shape.iter().any(|&dim| dim <= 0) {
            return Err(format!(
                "Inkling 2 layer {index}: mlp.experts.{name} must be [experts, out, in], got {weight_shape:?}"
            ));
        }
        if projections.len() == 2 && weight_shape[1] % 2 != 0 {
            return Err(format!(
                "Inkling 2 layer {index}: mlp.experts.{name} has an odd fused gate/up width {}",
                weight_shape[1]
            ));
        }
        let mut leaves: Vec<(&str, UniquePtr<MlxArray>)> = Vec::new();
        match scale {
            Some(scale) => {
                if mlxcel_core::array_itemsize(&weight) != 1 || weight_shape[2] % 4 != 0 {
                    return Err(format!(
                        "Inkling 2 layer {index}: packed mlp.experts.{name} must be bytes with a \
                         width divisible by 4, got {weight_shape:?}"
                    ));
                }
                let scale_shape = mlxcel_core::array_shape(&scale);
                if scale_shape.len() != 3
                    || scale_shape[0] != weight_shape[0]
                    || scale_shape[1] != weight_shape[1]
                    || scale_shape[2] <= 0
                {
                    return Err(format!(
                        "Inkling 2 layer {index}: mlp.experts.{name}_scale must be [{}, {}, groups], got {scale_shape:?}",
                        weight_shape[0], weight_shape[1]
                    ));
                }
                let packed = mlxcel_core::view(&weight, dtype::UINT32);
                leaves.push(("weight", packed));
                leaves.push(("scales", normalized_e4m3(scale)?));
            }
            None => leaves.push(("weight", weight)),
        }
        if amax.is_some() != denominator.is_some() {
            return Err(format!(
                "Inkling 2 layer {index}: mlp.experts.{name}_amax and _outer_scale_denominator must come together"
            ));
        }
        let global = match (amax, denominator) {
            (Some(amax), Some(denominator)) => {
                if leaves.len() != 2 {
                    return Err(format!(
                        "Inkling 2 layer {index}: mlp.experts.{name}_amax needs a packed plane with a _scale sidecar"
                    ));
                }
                let amax_shape = mlxcel_core::array_shape(&amax);
                if amax_shape != [weight_shape[0]] && amax_shape.iter().product::<i32>() != 1 {
                    return Err(format!(
                        "Inkling 2 layer {index}: mlp.experts.{name}_amax must be [{}] or a scalar, got {amax_shape:?}",
                        weight_shape[0]
                    ));
                }
                if mlxcel_core::array_size(&denominator) != 1 {
                    return Err(format!(
                        "Inkling 2 layer {index}: mlp.experts.{name}_outer_scale_denominator must be a scalar"
                    ));
                }
                let amax = mlxcel_core::astype(&amax, dtype::FLOAT32);
                let denominator =
                    mlxcel_core::reshape(&mlxcel_core::astype(&denominator, dtype::FLOAT32), &[1]);
                let scale2 = mlxcel_core::divide(&amax, &denominator);
                let scale2 = mlxcel_core::broadcast_to(&scale2, &[weight_shape[0]]);
                let global = mlxcel_core::multiply_scalar(&scale2, NVFP4_GLOBAL_SCALE_FACTOR);
                Some(mlxcel_core::contiguous(&global, false))
            }
            _ => None,
        };
        for (leaf, value) in leaves {
            if projections.len() == 2 {
                let (gate, up) = split_expert_w13(&value)?;
                insert(out, format!("{base}.gate_proj.{leaf}"), gate)?;
                insert(out, format!("{base}.up_proj.{leaf}"), up)?;
            } else {
                insert(out, format!("{base}.down_proj.{leaf}"), value)?;
            }
        }
        if let Some(global) = global {
            for projection in projections {
                insert(
                    out,
                    format!("{base}.{projection}.native_global_scale"),
                    mlxcel_core::copy(&global),
                )?;
            }
        }
    }
    if !parts.is_empty() {
        return Err(format!(
            "Inkling 2 layer {index}: unmapped mlp.experts tensors {:?}",
            parts.keys().collect::<Vec<_>>()
        ));
    }
    Ok(())
}

/// Fuse the two cross-pass query tensors (`[core, 2, h]` each, possibly
/// stored flat as `[2 * core, h]`) into `message_queries [core, 4, h]` in
/// attn, k, v, mlp order: `concat([qa[:, :1], kv, qa[:, 1:]], axis=1)`.
fn fuse_message_queries(
    mut cross_pass: BTreeMap<&'static str, UniquePtr<MlxArray>>,
    hidden_size: usize,
    out: &mut WeightMap,
) -> Result<(), String> {
    let (attn_mlp, kv) = match (cross_pass.remove("attn_mlp"), cross_pass.remove("kv")) {
        (None, None) => return Ok(()),
        (Some(attn_mlp), Some(kv)) => (attn_mlp, kv),
        _ => {
            return Err(
                "Inkling 2 checkpoint has only one of middle_cycle_cross_pass_queries and \
                 middle_cycle_cross_pass_kv_queries"
                    .into(),
            );
        }
    };
    let width = hidden_size as i32;
    let reshape = |value: &MlxArray, name: &str| -> Result<UniquePtr<MlxArray>, String> {
        let size = mlxcel_core::array_size(value) as i64;
        if size == 0 || size % (2 * i64::from(width)) != 0 {
            return Err(format!(
                "Inkling 2 {name} has {size} elements, not a multiple of 2 * hidden_size {width}"
            ));
        }
        let core = (size / (2 * i64::from(width))) as i32;
        Ok(mlxcel_core::reshape(value, &[core, 2, width]))
    };
    let attn_mlp = reshape(&attn_mlp, "middle_cycle_cross_pass_queries")?;
    let kv = reshape(&kv, "middle_cycle_cross_pass_kv_queries")?;
    if mlxcel_core::array_shape(&attn_mlp) != mlxcel_core::array_shape(&kv) {
        return Err("Inkling 2 cross-pass query tensors disagree on the core length".into());
    }
    let core = mlxcel_core::array_shape(&attn_mlp)[0];
    let attn = mlxcel_core::slice(&attn_mlp, &[0, 0, 0], &[core, 1, width]);
    let mlp = mlxcel_core::slice(&attn_mlp, &[0, 1, 0], &[core, 2, width]);
    let fused = mlxcel_core::concatenate_many(&[&attn, &kv, &mlp], 1);
    insert(out, format!("{TEXT_PREFIX}message_queries"), fused)
}

#[cfg(test)]
#[path = "sanitize_tests.rs"]
mod tests;
