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

use std::collections::HashMap;

use anyhow::{Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::dense::preprocess::conv_weight_to_channels_last;

#[derive(Default)]
struct Triple {
    query: Option<UniquePtr<MlxArray>>,
    key: Option<UniquePtr<MlxArray>>,
    value: Option<UniquePtr<MlxArray>>,
}

#[derive(Default)]
struct Pair {
    gate: Option<UniquePtr<MlxArray>>,
    up: Option<UniquePtr<MlxArray>>,
}

fn strip_model_prefix(mut key: &str) -> &str {
    for prefix in ["dinov2_with_registers.", "dinov2."] {
        if let Some(rest) = key.strip_prefix(prefix) {
            key = rest;
            break;
        }
    }
    key
}

fn embedding_key(key: &str) -> Option<&'static str> {
    match key {
        "embeddings.cls_token" => Some("cls_token"),
        "embeddings.mask_token" => Some("mask_token"),
        "embeddings.register_tokens" => Some("register_tokens"),
        "embeddings.position_embeddings" => Some("pos_embed"),
        "embeddings.patch_embeddings.projection.weight" => Some("patch_embed.proj.weight"),
        "embeddings.patch_embeddings.projection.bias" => Some("patch_embed.proj.bias"),
        "layernorm.weight" => Some("norm.weight"),
        "layernorm.bias" => Some("norm.bias"),
        _ => None,
    }
}

fn projection_part(rest: &str) -> Option<(usize, &str)> {
    let candidates = [
        ("attention.attention.query.", 0),
        ("attention.attention.key.", 1),
        ("attention.attention.value.", 2),
        ("attention.attention.q_proj.", 0),
        ("attention.attention.k_proj.", 1),
        ("attention.attention.v_proj.", 2),
        ("attention.q_proj.", 0),
        ("attention.k_proj.", 1),
        ("attention.v_proj.", 2),
    ];
    candidates
        .into_iter()
        .find_map(|(prefix, part)| rest.strip_prefix(prefix).map(|suffix| (part, suffix)))
}

fn swiglu_part(rest: &str) -> Option<(usize, &str)> {
    [("mlp.gate_proj.", 0), ("mlp.up_proj.", 1)]
        .into_iter()
        .find_map(|(prefix, part)| rest.strip_prefix(prefix).map(|suffix| (part, suffix)))
}

fn mapped_block_key(rest: &str) -> String {
    let mappings = [
        ("attention.output.dense.", "attn.proj."),
        ("attention.o_proj.", "attn.proj."),
        ("layer_scale1.lambda1", "ls1.gamma"),
        ("layer_scale2.lambda1", "ls2.gamma"),
        ("norm1.", "norm1."),
        ("norm2.", "norm2."),
        ("mlp.fc1.", "mlp.fc1."),
        ("mlp.fc2.", "mlp.fc2."),
        ("mlp.weights_in.", "mlp.w12."),
        ("mlp.weights_out.", "mlp.w3."),
        ("mlp.down_proj.", "mlp.w3."),
    ];
    for (from, to) in mappings {
        if let Some(suffix) = rest.strip_prefix(from) {
            return format!("{to}{suffix}");
        }
    }
    rest.to_string()
}

fn set_triple_part(triple: &mut Triple, part: usize, value: UniquePtr<MlxArray>) {
    match part {
        0 => triple.query = Some(value),
        1 => triple.key = Some(value),
        _ => triple.value = Some(value),
    }
}

fn set_pair_part(pair: &mut Pair, part: usize, value: UniquePtr<MlxArray>) {
    if part == 0 {
        pair.gate = Some(value);
    } else {
        pair.up = Some(value);
    }
}

fn concatenate_triple(key: &str, mut parts: Triple) -> Result<UniquePtr<MlxArray>> {
    let query = parts
        .query
        .take()
        .ok_or_else(|| anyhow::anyhow!("DINOv2 checkpoint is missing query part for {key}"))?;
    let key_part = parts
        .key
        .take()
        .ok_or_else(|| anyhow::anyhow!("DINOv2 checkpoint is missing key part for {key}"))?;
    let value = parts
        .value
        .take()
        .ok_or_else(|| anyhow::anyhow!("DINOv2 checkpoint is missing value part for {key}"))?;
    Ok(mlxcel_core::concatenate_many(
        &[&query, &key_part, &value],
        0,
    ))
}

fn concatenate_pair(key: &str, mut parts: Pair) -> Result<UniquePtr<MlxArray>> {
    let gate = parts.gate.take().ok_or_else(|| {
        anyhow::anyhow!("DINOv2 checkpoint is missing SwiGLU gate part for {key}")
    })?;
    let up = parts
        .up
        .take()
        .ok_or_else(|| anyhow::anyhow!("DINOv2 checkpoint is missing SwiGLU up part for {key}"))?;
    Ok(mlxcel_core::concatenate(&gate, &up, 0))
}

/// Normalize HF standard/register checkpoints and already-converted DINOv2 maps.
pub(super) fn sanitize_weights(weights: WeightMap) -> Result<WeightMap> {
    let mut output = WeightMap::with_capacity(weights.len());
    let mut qkv: HashMap<String, Triple> = HashMap::new();
    let mut swiglu: HashMap<String, Pair> = HashMap::new();

    for (raw_key, mut value) in weights {
        let key = strip_model_prefix(&raw_key);
        if key.starts_with("classifier.") {
            continue;
        }
        if let Some(mapped) = embedding_key(key) {
            if mapped == "patch_embed.proj.weight" {
                value = conv_weight_to_channels_last(&value, 3)?;
            }
            output.insert(mapped.into(), value);
            continue;
        }
        if key == "patch_embed.proj.weight" {
            value = conv_weight_to_channels_last(&value, 3)?;
            output.insert(key.into(), value);
            continue;
        }

        let Some(block_rest) = key.strip_prefix("encoder.layer.") else {
            output.insert(key.into(), value);
            continue;
        };
        let Some((index, rest)) = block_rest.split_once('.') else {
            bail!("malformed DINOv2 encoder key `{raw_key}`");
        };
        if let Some((part, suffix)) = projection_part(rest) {
            let mapped = format!("blocks.{index}.attn.qkv.{suffix}");
            set_triple_part(qkv.entry(mapped).or_default(), part, value);
            continue;
        }
        if let Some((part, suffix)) = swiglu_part(rest) {
            let mapped = format!("blocks.{index}.mlp.w12.{suffix}");
            set_pair_part(swiglu.entry(mapped).or_default(), part, value);
            continue;
        }
        output.insert(format!("blocks.{index}.{}", mapped_block_key(rest)), value);
    }

    for (key, parts) in qkv {
        output.insert(key.clone(), concatenate_triple(&key, parts)?);
    }
    for (key, parts) in swiglu {
        output.insert(key.clone(), concatenate_pair(&key, parts)?);
    }
    Ok(output)
}

/// Copy one DINOv2 backbone scope out of a larger dense-model checkpoint.
///
/// `prefix` is the parent path without a trailing dot. An empty prefix selects
/// the complete map. The returned keys have the scope removed before normal
/// Hugging Face DINOv2 key conversion is applied.
pub(super) fn sanitize_scoped_weights(weights: &WeightMap, prefix: &str) -> Result<WeightMap> {
    let prefix = prefix.trim_end_matches('.');
    let dotted = (!prefix.is_empty()).then(|| format!("{prefix}."));
    let mut scoped = WeightMap::new();
    for (key, value) in weights {
        let scoped_key = match &dotted {
            Some(prefix) => key.strip_prefix(prefix),
            None => Some(key.as_str()),
        };
        if let Some(scoped_key) = scoped_key {
            scoped.insert(scoped_key.to_owned(), mlxcel_core::copy(value));
        }
    }
    if scoped.is_empty() {
        bail!("DINOv2 checkpoint has no weights under scope `{prefix}`");
    }
    sanitize_weights(scoped)
}
