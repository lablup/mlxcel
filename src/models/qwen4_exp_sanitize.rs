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

//! Qwen4-Exp checkpoint key canonicalization, mirroring upstream
//! `Model.sanitize` (`mlx_vlm/models/qwen4_exp/qwen4_exp.py`).
//!
//! Canonical text keys are `model.*` and `lm_head.*`, vision keys are
//! `vision_tower.*`. Hugging Face (`model.language_model.*`, `model.visual.*`),
//! mlx-vlm (`language_model.model.*`) and mlx-lm/OptiQ (`model.*`) layouts all
//! map onto them.

use std::collections::BTreeSet;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::ResolvedQwen4ExpConfig;

pub struct SanitizedWeights {
    pub text: WeightMap,
    pub vision: WeightMap,
}

pub(crate) fn canonical_key(key: &str) -> String {
    let key = if let Some(rest) = key.strip_prefix("model.language_model.visual.") {
        format!("vision_tower.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.lm_head") {
        format!("lm_head{rest}")
    } else if let Some(rest) = key.strip_prefix("model.visual.") {
        format!("vision_tower.{rest}")
    } else {
        key.to_string()
    };
    rename_ngram_shard(&key)
}

/// `.ngram_embedding.shard_N` -> `.ngram_embedding.shards.N`, as a tensor
/// key (`...shard_N.weight`) or a quantization module path (`...shard_N`).
fn rename_ngram_shard(key: &str) -> String {
    const MARKER: &str = ".ngram_embedding.shard_";
    let Some(start) = key.find(MARKER) else {
        return key.to_string();
    };
    let digits_start = start + MARKER.len();
    let digits_len = key[digits_start..]
        .bytes()
        .take_while(u8::is_ascii_digit)
        .count();
    let rest = &key[digits_start + digits_len..];
    if digits_len == 0 || !(rest.is_empty() || rest.starts_with('.')) {
        return key.to_string();
    }
    format!(
        "{}.ngram_embedding.shards.{}{}",
        &key[..start],
        &key[digits_start..digits_start + digits_len],
        &key[digits_start + digits_len..]
    )
}

/// Canonicalize keys, drop what the base model does not use, and bring the
/// expert and convolution tensors into the layout the loader reads.
pub fn sanitize_weights(
    weights: WeightMap,
    config: &ResolvedQwen4ExpConfig,
) -> Result<SanitizedWeights, String> {
    if let Some(key) = weights.keys().find(|key| key.ends_with("weight_scale_inv")) {
        return Err(format!(
            "Qwen4-Exp FP8 checkpoints (block-scaled `weight_scale_inv` tensors, e.g. `{key}`) \
             are not supported; convert the checkpoint to an MLX quantization first"
        ));
    }

    let mut text = WeightMap::new();
    let mut vision = WeightMap::new();
    for (key, value) in weights {
        // The MTP predictor is a separate speculative artifact.
        if key.starts_with("mtp.") {
            continue;
        }
        let key = canonical_key(&key);
        if key.starts_with("model.mtp.") {
            continue;
        }
        if key.starts_with("vision_tower.") {
            vision.insert(key, value);
            continue;
        }
        if config.raw.tie_word_embeddings && key.starts_with("lm_head.") {
            continue;
        }
        let value = if key.ends_with("conv1d.weight") && conv_needs_moveaxis(&value) {
            mlxcel_core::moveaxis(&value, 2, 1)
        } else {
            value
        };
        text.insert(key, value);
    }

    // Visit the layers whose experts the checkpoint stores rather than
    // `0..num_hidden_layers`: the config count is unbounded until the loader
    // fails on the first missing layer.
    let expert_blocks: BTreeSet<String> = text
        .keys()
        .filter_map(|key| {
            let (layer, rest) = key.strip_prefix("model.layers.")?.split_once('.')?;
            rest.starts_with("mlp.experts.")
                .then(|| format!("model.layers.{layer}.mlp"))
        })
        .collect();
    for prefix in &expert_blocks {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            stack_per_expert(&mut text, prefix, projection, config.raw.num_experts)?;
        }
        split_fused_gate_up(&mut text, prefix)?;
    }
    Ok(SanitizedWeights { text, vision })
}

fn conv_needs_moveaxis(value: &MlxArray) -> bool {
    let shape = mlxcel_core::array_shape(value);
    shape.len() == 3 && shape[2] != 1
}

const EXPERT_SUFFIXES: [&str; 4] = ["weight", "scales", "biases", "bias"];

/// `{prefix}.experts.{e}.{projection}.{suffix}` for every expert ->
/// `{prefix}.switch_mlp.{projection}.{suffix}` stacked on a leading expert axis.
fn stack_per_expert(
    weights: &mut WeightMap,
    prefix: &str,
    projection: &str,
    num_experts: usize,
) -> Result<(), String> {
    for suffix in EXPERT_SUFFIXES {
        if !weights.contains_key(&format!("{prefix}.experts.0.{projection}.{suffix}")) {
            continue;
        }
        let mut parts: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut first_shape = None;
        for expert in 0..num_experts {
            let key = format!("{prefix}.experts.{expert}.{projection}.{suffix}");
            let part = weights
                .remove(&key)
                .ok_or_else(|| format!("Qwen4-Exp checkpoint is missing {key}"))?;
            let shape = mlxcel_core::array_shape(&part);
            match &first_shape {
                None => first_shape = Some(shape),
                Some(first) if *first != shape => {
                    return Err(format!(
                        "{key} has shape {shape:?}, but expert 0 has {first:?}: \
                         the experts cannot be stacked"
                    ));
                }
                Some(_) => {}
            }
            parts.push(part);
        }
        weights.insert(
            format!("{prefix}.switch_mlp.{projection}.{suffix}"),
            mlxcel_core::utils::stack_arrays(&parts, 0),
        );
    }
    Ok(())
}

/// Fused `{prefix}.experts.gate_up_proj` (`[experts, 2 * intermediate, hidden]`)
/// -> `switch_mlp.gate_proj` / `switch_mlp.up_proj`, and
/// `{prefix}.experts.down_proj` -> `switch_mlp.down_proj`.
fn split_fused_gate_up(weights: &mut WeightMap, prefix: &str) -> Result<(), String> {
    let source = format!("{prefix}.experts.gate_up_proj");
    let has_fused =
        weights.contains_key(&source) || weights.contains_key(&format!("{source}.weight"));
    if !has_fused {
        return Ok(());
    }
    for name in ["gate_up_proj", "down_proj"] {
        let key = format!("{prefix}.experts.{name}");
        if let Some(scales) = weights.remove(&format!("{key}_scales")) {
            weights.insert(format!("{key}.scales"), scales);
        }
        if let Some(bare) = weights.remove(&key) {
            weights.insert(format!("{key}.weight"), bare);
        }
    }

    let weight_key = format!("{source}.weight");
    let rank = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::array_shape(w).len())
        .unwrap_or(0);
    if rank != 3 {
        return Err(format!(
            "{source} must be [num_experts, output_rows, input_columns], got rank {rank}"
        ));
    }
    for suffix in EXPERT_SUFFIXES {
        let Some(value) = weights.remove(&format!("{source}.{suffix}")) else {
            continue;
        };
        let shape = mlxcel_core::array_shape(&value);
        let rank = if suffix == "bias" { 2 } else { 3 };
        if shape.len() != rank {
            return Err(format!(
                "{source}.{suffix} must be [num_experts, output_rows{}], got shape {shape:?}",
                if rank == 3 { ", input_columns" } else { "" }
            ));
        }
        let axis = 1;
        let rows = shape[axis];
        if rows % 2 != 0 {
            return Err(format!(
                "{source}.{suffix} has an odd fused row count {rows}"
            ));
        }
        for (half, target) in ["gate_proj", "up_proj"].into_iter().enumerate() {
            let mut start = vec![0; shape.len()];
            let mut stop = shape.clone();
            start[axis] = half as i32 * (rows / 2);
            stop[axis] = (half as i32 + 1) * (rows / 2);
            let part = mlxcel_core::slice(&value, &start, &stop);
            weights.insert(
                format!("{prefix}.switch_mlp.{target}.{suffix}"),
                mlxcel_core::contiguous(&part, false),
            );
        }
    }
    for suffix in EXPERT_SUFFIXES {
        if let Some(value) = weights.remove(&format!("{prefix}.experts.down_proj.{suffix}")) {
            weights.insert(format!("{prefix}.switch_mlp.down_proj.{suffix}"), value);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4_exp_sanitize_tests.rs"]
mod tests;
