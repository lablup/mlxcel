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

//! MiMo-V2 checkpoint remapping: fused-QKV unfuse, FP8 dequantization, MXFP4
//! expert repacking, and per-expert stacking into the Flash loader's layout.
//! The family documentation lives in [`super::mimo_v2`].

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::fp8_block::reconstruct_block_fp8_tensor;
use super::mimo_v2_flash::ModelArgs;

/// FP8 block extent of the vendor `weight_scale_inv` grids.
const FP8_BLOCK: usize = 128;

/// Layer index of a `model.layers.<i>.…` style key, the way upstream's
/// `_layer_index` reads it.
fn layer_index(key: &str) -> Option<usize> {
    let mut parts = key.split('.');
    while let Some(part) = parts.next() {
        if part == "layers" {
            return parts.next()?.parse().ok();
        }
    }
    None
}

/// Tensor-parallel degrees consistent with one fused tensor's row count and
/// scale-grid row count. Port of upstream `_candidate_degrees`.
pub(crate) fn candidate_degrees(
    rows: i32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    grid_rows: i32,
) -> Vec<usize> {
    let mut found = Vec::new();
    let mut degree = 1usize;
    while degree <= kv_heads {
        if heads.is_multiple_of(degree) && kv_heads.is_multiple_of(degree) {
            let shard = ((heads / degree) * head_dim
                + (kv_heads / degree) * (head_dim + v_head_dim)) as i32;
            let block = FP8_BLOCK as i32;
            let blocks = (shard + block - 1) / block;
            if shard * degree as i32 == rows && blocks * degree as i32 == grid_rows {
                found.push(degree);
            }
        }
        degree *= 2;
    }
    found
}

/// Per-layer head geometry, as upstream `_qkv_heads` reads it.
fn qkv_heads(args: &ModelArgs, layer: usize) -> (usize, usize, usize, usize) {
    if args.is_sliding_layer(layer) {
        (
            args.swa_num_attention_heads,
            args.swa_num_key_value_heads,
            args.swa_head_dim,
            args.swa_v_head_dim,
        )
    } else {
        (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
            args.v_head_dim,
        )
    }
}

fn is_mtp_key(key: &str) -> bool {
    key.contains(".mtp.")
}

fn is_fused_qkv_weight(key: &str) -> bool {
    key.ends_with("self_attn.qkv_proj.weight") && !is_mtp_key(key)
}

fn is_fused_qkv_scale(key: &str) -> bool {
    key.ends_with("self_attn.qkv_proj.weight_scale_inv") && !is_mtp_key(key)
}

/// Recover the tensor-parallel degree the checkpoint's fused QKV weights were
/// sharded with.
///
/// A layer whose per-shard rows are a whole number of 128-row blocks admits
/// several degrees on its own, so the degree is intersected over every fused
/// tensor that carries a scale grid. Port of upstream
/// `_tensor_parallel_degree`.
pub(crate) fn tensor_parallel_degree(
    weights: &WeightMap,
    args: &ModelArgs,
) -> Result<usize, String> {
    let mut common: Option<Vec<usize>> = None;
    for key in weights.keys() {
        if !is_fused_qkv_weight(key) {
            continue;
        }
        let Some(weight) = weights.get(key) else {
            continue;
        };
        let Some(scale) = weights.get(&format!("{key}_scale_inv")) else {
            continue;
        };
        let layer = layer_index(key)
            .ok_or_else(|| format!("MiMo-V2 fused QKV key {key:?} has no layer index"))?;
        let (heads, kv_heads, head_dim, v_head_dim) = qkv_heads(args, layer);
        let shape = mlxcel_core::array_shape(weight);
        let scale_shape = mlxcel_core::array_shape(scale);
        let rows = *shape
            .first()
            .ok_or_else(|| format!("MiMo-V2 fused QKV weight {key:?} has no rows"))?;
        let grid_rows = *scale_shape
            .first()
            .ok_or_else(|| format!("MiMo-V2 fused QKV scale grid {key:?} has no rows"))?;
        let found = candidate_degrees(rows, heads, kv_heads, head_dim, v_head_dim, grid_rows);
        common = Some(match common {
            Some(previous) => previous
                .into_iter()
                .filter(|degree| found.contains(degree))
                .collect(),
            None => found,
        });
    }
    match common {
        Some(degrees) if !degrees.is_empty() => Ok(degrees.into_iter().max().unwrap_or(1)),
        _ => Err(
            "no tensor-parallel degree explains the fused qkv shapes: the row counts and the \
             padded FP8 scale grids of self_attn.qkv_proj.weight do not agree on a shard count"
                .to_string(),
        ),
    }
}

fn dequant_block_fp8(
    weight: &MlxArray,
    scale: &MlxArray,
    lut: &MlxArray,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    let dense = reconstruct_block_fp8_tensor(name, weight, scale, FP8_BLOCK, lut)?;
    Ok(mlxcel_core::astype(&dense, dtype::BFLOAT16))
}

/// Split every fused `self_attn.qkv_proj.weight` back into `q_proj`/`k_proj`/
/// `v_proj`, dequantizing each tensor-parallel shard with its own slice of the
/// FP8 scale grid. Port of upstream `_unfuse_qkv`.
///
/// Takes ownership so a non-fused tensor is moved, not copied: the native
/// checkpoint is 173.8 GB and a lazy copy of every untouched plane would double
/// the load's resident set.
pub(crate) fn unfuse_qkv(weights: WeightMap, args: &ModelArgs) -> Result<WeightMap, String> {
    let sharded = weights.keys().any(|key| is_fused_qkv_scale(key));
    let degree = if sharded {
        tensor_parallel_degree(&weights, args)?
    } else {
        1
    };

    let lut = {
        let table: Vec<f32> = (0..=u8::MAX).map(super::sanitize::f8_e4m3_to_f32).collect();
        mlxcel_core::from_slice_f32(&table, &[256])
    };

    let mut source = weights;
    let keys: Vec<String> = source.keys().cloned().collect();
    let mut out = WeightMap::new();
    for key in keys {
        if is_fused_qkv_scale(&key) {
            source.remove(&key);
            continue;
        }
        let Some(value) = source.remove(&key) else {
            continue;
        };
        if !is_fused_qkv_weight(&key) {
            out.insert(key, value);
            continue;
        }

        let scale = source.remove(&format!("{key}_scale_inv"));
        let prefix = key
            .strip_suffix("qkv_proj.weight")
            .ok_or_else(|| format!("MiMo-V2 fused key {key:?} has no qkv_proj suffix"))?;
        let layer = layer_index(&key)
            .ok_or_else(|| format!("MiMo-V2 fused QKV key {key:?} has no layer index"))?;
        let (heads, kv_heads, head_dim, v_head_dim) = qkv_heads(args, layer);
        let q_rows = heads / degree * head_dim;
        let k_rows = kv_heads / degree * head_dim;
        let v_rows = kv_heads / degree * v_head_dim;
        let shard = q_rows + k_rows + v_rows;
        let blocks = shard.div_ceil(FP8_BLOCK) as i32;
        let shard_i32 = i32::try_from(shard)
            .map_err(|_| format!("MiMo-V2 fused QKV shard {key:?} exceeds i32 rows"))?;
        let cols = *mlxcel_core::array_shape(&value)
            .get(1)
            .ok_or_else(|| format!("MiMo-V2 fused QKV weight {key:?} is not rank 2"))?;

        let mut parts: [Vec<UniquePtr<MlxArray>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for rank in 0..degree {
            let start = rank as i32 * shard_i32;
            let rows = mlxcel_core::slice(&value, &[start, 0], &[start + shard_i32, cols]);
            let dense = match scale.as_ref() {
                Some(grid) => {
                    let grid_start = rank as i32 * blocks;
                    let grid = mlxcel_core::slice(
                        grid,
                        &[grid_start, 0],
                        &[grid_start + blocks, mlxcel_core::array_shape(grid)[1]],
                    );
                    dequant_block_fp8(&rows, &grid, &lut, &key)?
                }
                None => rows,
            };
            let mut row = 0i32;
            for (part, size) in parts
                .iter_mut()
                .zip([q_rows as i32, k_rows as i32, v_rows as i32])
            {
                part.push(mlxcel_core::slice(&dense, &[row, 0], &[row + size, cols]));
                row += size;
            }
        }

        for (leaf, chunks) in ["q_proj", "k_proj", "v_proj"].into_iter().zip(parts) {
            let refs: Vec<&MlxArray> = chunks
                .iter()
                .map(|chunk| chunk.as_ref().expect("slice returns an array"))
                .collect();
            out.insert(
                format!("{prefix}{leaf}.weight"),
                mlxcel_core::concatenate_many(&refs, 0),
            );
        }
    }
    Ok(out)
}

/// Dequantize every remaining block-FP8 projection (`weight_scale_inv`).
///
/// The native release stores the dense prefix MLP and (for V2.5) the routed
/// experts as E4M3 bytes with a 128x128 inverse-scale grid. The fused QKV
/// tensors are unfused first, with their per-shard grids; everything still
/// carrying a `_scale_inv` sidecar after that is a plain projection.
///
/// Takes ownership so a checkpoint with no FP8 tensors pays no copy at all (the
/// converted mlx-vlm release is already dequantized).
pub(crate) fn dequantize_fp8_weights(weights: WeightMap) -> Result<WeightMap, String> {
    if !weights
        .keys()
        .any(|key| key.ends_with("_scale_inv") && !is_mtp_key(key))
    {
        return Ok(weights);
    }
    let lut = {
        let table: Vec<f32> = (0..=u8::MAX).map(super::sanitize::f8_e4m3_to_f32).collect();
        mlxcel_core::from_slice_f32(&table, &[256])
    };
    let mut source = weights;
    let mut out = WeightMap::new();
    let scale_keys: Vec<String> = source
        .keys()
        .filter(|key| key.ends_with("_scale_inv") && !is_mtp_key(key))
        .cloned()
        .collect();
    for key in scale_keys {
        let Some(base) = key.strip_suffix("_scale_inv").map(str::to_string) else {
            continue;
        };
        let Some(scale) = source.remove(&key) else {
            continue;
        };
        let weight = source.remove(&base).ok_or_else(|| {
            format!(
                "MiMo-V2 {key} has no matching weight at {base}; the FP8 sidecar cannot be applied"
            )
        })?;
        let dense = reconstruct_block_fp8_tensor(&base, &weight, &scale, FP8_BLOCK, &lut)?;
        out.insert(base, mlxcel_core::astype(&dense, dtype::BFLOAT16));
    }
    for (key, value) in source {
        out.insert(key, value);
    }
    Ok(out)
}

const EXPERT_PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

fn expert_count(weights: &WeightMap, root: &str) -> usize {
    let prefix = format!("{root}.");
    let mut highest = 0usize;
    let mut found = false;
    for key in weights.keys() {
        let Some(rest) = key.strip_prefix(&prefix) else {
            continue;
        };
        let Some(index) = rest.split('.').next() else {
            continue;
        };
        if let Ok(index) = index.parse::<usize>() {
            found = true;
            highest = highest.max(index + 1);
        }
    }
    if found { highest } else { 0 }
}

/// Stack the native per-expert layout into the Flash loader's `switch_mlp`
/// planes.
///
/// Upstream's Flash sanitizer stacks `mlp.experts.{e}.{proj}.{leaf}` into
/// `mlp.switch_mlp.{proj}.{leaf}`. The MiMo-V2 release ships `weight` (and,
/// after [`convert_mxfp4_experts`] or [`dequantize_fp8_weights`], `scales`);
/// stacking here means the shared expert loader reads one plan per projection.
pub(crate) fn stack_experts(weights: WeightMap, layers: usize) -> Result<WeightMap, String> {
    let mut out = weights;
    for layer in 0..layers {
        let root = format!("model.layers.{layer}.mlp.experts");
        let experts = expert_count(&out, &root);
        if experts == 0 {
            continue;
        }
        for projection in EXPERT_PROJECTIONS {
            for leaf in ["weight", "scales", "biases"] {
                let first = format!("{root}.0.{projection}.{leaf}");
                if !out.contains_key(&first) {
                    continue;
                }
                let mut parts = Vec::with_capacity(experts);
                for expert in 0..experts {
                    let key = format!("{root}.{expert}.{projection}.{leaf}");
                    parts.push(out.remove(&key).ok_or_else(|| {
                        format!(
                            "MiMo-V2 has {experts} routed experts for layer {layer} but is missing {key}"
                        )
                    })?);
                }
                out.insert(
                    format!("model.layers.{layer}.mlp.switch_mlp.{projection}.{leaf}"),
                    mlxcel_core::stack_owned(&parts, 0),
                );
            }
        }
        let leftovers: Vec<String> = out
            .keys()
            .filter(|key| key.starts_with(&format!("{root}.")))
            .cloned()
            .collect();
        for key in leftovers {
            out.remove(&key);
        }
    }
    Ok(out)
}

/// Rename and reinterpret the native MXFP4 routed experts.
///
/// Upstream `_convert_mxfp4_experts`: a `<plane>.weight` with a sibling
/// `<plane>.weight_scale` becomes `uint32` `<plane>.weight` plus
/// `<plane>.scales`, and the sidecar key is dropped. Takes ownership so an
/// already-converted checkpoint is passed through without copying.
pub(crate) fn convert_mxfp4_experts(mut weights: WeightMap) -> WeightMap {
    let mut scales = WeightMap::new();
    let scale_keys: Vec<String> = weights
        .keys()
        .filter(|key| key.ends_with(".weight_scale"))
        .cloned()
        .collect();
    for key in scale_keys {
        if let Some(value) = weights.remove(&key) {
            scales.insert(key, value);
        }
    }
    let mut out = weights;
    let keys: Vec<String> = out.keys().cloned().collect();
    for key in keys {
        let Some(value) = out.remove(&key) else {
            continue;
        };
        match scales.remove(&format!("{key}_scale")) {
            Some(scale) => {
                out.insert(key.clone(), mlxcel_core::view(&value, dtype::UINT32));
                out.insert(format!("{}.scales", key.trim_end_matches(".weight")), scale);
            }
            None => {
                out.insert(key, value);
            }
        }
    }
    out
}

pub(crate) fn has_mxfp4_expert_weights(weights: &WeightMap) -> bool {
    weights.keys().any(|key| key.ends_with(".weight_scale"))
}

/// Full MiMo-V2 text-weight normalization: drop the modalities this loader
/// does not consume (audio, MTP), strip the `language_model.` / `visual.`
/// checkpoint prefixes, repack MXFP4 experts, then unfuse QKV.
pub(crate) fn sanitize_text_weights(
    weights: WeightMap,
    args: &ModelArgs,
) -> Result<WeightMap, String> {
    let mut normalized = WeightMap::new();
    for (key, value) in weights {
        if key.starts_with("audio_encoder.") || key.starts_with("speech_embeddings.") {
            continue;
        }
        if key.starts_with("vision_tower.") {
            continue;
        }
        let key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else if let Some(stripped) = key.strip_prefix("visual.") {
            // The native release stores the tower under `visual.`; the wrapper
            // reads it under `vision_tower.`.
            format!("vision_tower.{stripped}")
        } else {
            key
        };
        if is_mtp_key(&key) || key.starts_with("model.mtp") {
            continue;
        }
        normalized.insert(key, value);
    }

    let normalized = if has_mxfp4_expert_weights(&normalized) {
        convert_mxfp4_experts(normalized)
    } else {
        normalized
    };

    // Unfuse first: it consumes the qkv `_scale_inv` grids per shard. Only the
    // projections left after that are plain FP8 tensors.
    let normalized = if normalized.keys().any(|key| is_fused_qkv_weight(key)) {
        unfuse_qkv(normalized, args)?
    } else {
        normalized
    };
    let normalized = dequantize_fp8_weights(normalized)?;
    stack_experts(normalized, args.num_hidden_layers)
}

/// Split off the vision-tower half of a raw checkpoint, normalizing the
/// `visual.` prefix to `vision_tower.`.
pub(crate) fn take_vision_weights(weights: &mut WeightMap) -> WeightMap {
    let keys: Vec<String> = weights
        .keys()
        .filter(|key| key.starts_with("vision_tower.") || key.starts_with("visual."))
        .cloned()
        .collect();
    let mut vision = WeightMap::new();
    for key in keys {
        if let Some(value) = weights.remove(&key) {
            let normalized = key
                .strip_prefix("visual.")
                .map(|rest| format!("vision_tower.{rest}"))
                .unwrap_or(key);
            vision.insert(normalized, value);
        }
    }
    vision
}
