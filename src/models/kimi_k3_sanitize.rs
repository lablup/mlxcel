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

//! Weight-key rules for Kimi K3 checkpoints. Idempotent: running
//! [`sanitize_weights`] on its own output is a no-op.
//!
//! Input is the published `moonshotai/Kimi-K3` layout (compressed-tensors
//! `mxfp4-pack-quantized` experts under `language_model.`), or a layer
//! truncated local copy of it, or an MLX-style conversion with per-expert
//! `.weight` / `.scales` / `.biases` planes. Output is what
//! `KimiK3Model::from_weights` reads:
//!
//! - `language_model.` stripped; every key that is not then under `model.`
//!   or `lm_head.` is dropped (`vision_tower.*`, `mm_projector.*`, processor
//!   buffers). The vision tower is #1342's.
//! - `model.mtp*` dropped (MTP is out of scope); `model.layers.N.*` with
//!   `N >= num_hidden_layers` dropped (layer-truncated copies).
//! - Legacy residual spellings `self_attention_res.proj_weight` /
//!   `.norm_weight` (and `mlp_res`, `output_attn_res`) renamed to
//!   `_proj.weight` / `_norm.weight`.
//! - KDA: `{q,k,v}_conv1d.weight [C, 1, K] -> [C, K, 1]`, the three concatenated
//!   into `qkv_conv.conv.weight [3P, K, 1]`; `{q,k,v}_proj` concatenated
//!   along axis 0 into `qkv_proj [3P, hidden]`; `A_log` sliced to the first
//!   `num_heads` entries; `dt_bias` flattened. Dtypes are left alone (`A_log`,
//!   `dt_bias`, `o_norm.weight` stay float32).
//! - MLA: `kv_b_proj.weight [H * (nope + v), rank]` decomposed into the
//!   per-head `embed_q` / `unembed_out` pair through
//!   `kimi_linear::decompose_kv_b_proj`.
//! - MoE: `block_sparse_moe.*` renamed to `mlp.*`,
//!   `gate.e_score_correction_bias` to `mlp.e_score_correction_bias`; the
//!   per-expert `w1` / `w3` / `w2` planes stacked into
//!   `mlp.switch_mlp.{gate,up,down}_proj`. A compressed-tensors
//!   `weight_packed` (uint8 `[out, in / 2]`) stack is reinterpreted as uint32
//!   `[E, out, in / 8]`, MLX's mxfp4 layout (8 E2M1 codes per uint32, low
//!   nibble first), with the uint8 E8M0 `weight_scale` stack as `.scales` and
//!   no biases. Planes are stacked one at a time, evaluated, and their
//!   per-expert sources released before the next plane, so the peak is one
//!   stacked plane rather than two copies of the layer.

use super::KimiK3TextConfig;
use crate::models::kimi_linear::decompose_kv_b_proj;
use mlxcel_core::dtype;
use mlxcel_core::utils::stack_arrays;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const TEXT_PREFIX: &str = "language_model.";

/// Legacy residual spellings and their canonical forms, as `(old leaf, new
/// leaf)` pairs applied to the `self_attention_res`, `mlp_res` and
/// `output_attn_res` stems.
const RESIDUAL_STEMS: [&str; 3] = ["self_attention_res", "mlp_res", "output_attn_res"];
const RESIDUAL_LEAVES: [(&str, &str); 2] = [
    ("proj_weight", "_proj.weight"),
    ("norm_weight", "_norm.weight"),
];

/// Map a raw checkpoint key to its text-model key, or `None` to drop it.
fn canonical_key(key: &str, num_hidden_layers: usize) -> Option<String> {
    let key = key.strip_prefix(TEXT_PREFIX).unwrap_or(key);
    if !(key.starts_with("model.") || key.starts_with("lm_head.")) {
        return None;
    }
    if key.starts_with("model.mtp") {
        return None;
    }
    if let Some(rest) = key.strip_prefix("model.layers.") {
        let layer: Option<usize> = rest.split('.').next().and_then(|n| n.parse().ok());
        match layer {
            Some(n) if n >= num_hidden_layers => return None,
            _ => {}
        }
    }
    for stem in RESIDUAL_STEMS {
        for (old_leaf, new_leaf) in RESIDUAL_LEAVES {
            let legacy = format!("{stem}.{old_leaf}");
            if let Some(head) = key.strip_suffix(&legacy) {
                return Some(format!("{head}{stem}{new_leaf}"));
            }
        }
    }
    Some(key.to_string())
}

fn eval_owned(array: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::eval(&array);
    array
}

/// Move every `{src}.{name}.{suffix}` to `{dst}.{name}.{suffix}`.
fn move_planes(weights: &mut WeightMap, src: &str, dst: &str, name: &str) {
    for suffix in ["weight", "scales", "biases"] {
        if let Some(w) = weights.remove(&format!("{src}.{name}.{suffix}")) {
            weights.insert(format!("{dst}.{name}.{suffix}"), w);
        }
    }
}

/// Reject a concatenation [`fuse_axis0`] would abort on: planes whose ranks
/// disagree, or whose extents differ on any axis other than the concatenated
/// one.
///
/// `concatenate` is the third MLX call in this file that takes the process
/// down on a bad argument instead of returning (see [`stack_expert_plane`] for
/// `stack` and `view`), and its arguments here are checkpoint data: a
/// partially converted or truncated q / k / v plane set reaches it with
/// mismatched shapes and aborts during sanitize, before any of the load-time
/// checks in `kimi_k3.rs` can name the tensor.
fn check_concat_compatible(sources: &[UniquePtr<MlxArray>], keys: &[String]) -> Result<(), String> {
    let Some(first) = sources.first() else {
        return Ok(());
    };
    let expected = mlxcel_core::array_shape(first);
    for (w, key) in sources.iter().zip(keys.iter()).skip(1) {
        let shape = mlxcel_core::array_shape(w);
        let compatible = shape.len() == expected.len()
            && shape
                .iter()
                .zip(expected.iter())
                .skip(1)
                .all(|(a, b)| a == b);
        if !compatible {
            return Err(format!(
                "{key}: shape {shape:?} cannot be concatenated on axis 0 with {}'s {expected:?}; \
                 every axis but the first must match",
                keys[0]
            ));
        }
    }
    Ok(())
}

/// Concatenate `{prefix}.{part}.{suffix}` for the given parts along axis 0
/// into `{prefix}.{fused}.{suffix}` when every part is present. Leaves the
/// map untouched (and returns `Ok(false)`) when the first part is absent,
/// which is the already-fused case.
fn fuse_axis0(
    weights: &mut WeightMap,
    prefix: &str,
    parts: &[&str],
    fused: &str,
    suffix: &str,
) -> Result<bool, String> {
    let keys: Vec<String> = parts
        .iter()
        .map(|p| format!("{prefix}.{p}.{suffix}"))
        .collect();
    if !weights.contains_key(&keys[0]) {
        return Ok(false);
    }
    let mut sources = Vec::with_capacity(keys.len());
    for key in &keys {
        let w = weights
            .remove(key)
            .ok_or_else(|| format!("{prefix}: cannot fuse {fused}.{suffix}, missing {key}"))?;
        sources.push(w);
    }
    check_concat_compatible(&sources, &keys)?;
    let refs: Vec<&MlxArray> = sources.iter().map(|w| w.as_ref().unwrap()).collect();
    let fused_array = eval_owned(mlxcel_core::concatenate_many(&refs, 0));
    weights.insert(format!("{prefix}.{fused}.{suffix}"), fused_array);
    Ok(true)
}

fn sanitize_kda_layer(
    weights: &mut WeightMap,
    attn_prefix: &str,
    config: &KimiK3TextConfig,
) -> Result<(), String> {
    // `[C, 1, K]` (PyTorch depthwise) -> `[C, K, 1]` (MLX). Only a weight whose
    // last axis is not 1 is transposed, so an already-sanitized weight is a
    // no-op.
    for part in ["q", "k", "v"] {
        let key = format!("{attn_prefix}.{part}_conv1d.weight");
        if let Some(w) = weights.get(&key) {
            let shape = mlxcel_core::array_shape(w);
            if shape.len() == 3 && shape[2] != 1 {
                let moved = mlxcel_core::swap_axes(w, 1, 2);
                weights.insert(key, moved);
            }
        }
    }
    if fuse_axis0(
        weights,
        attn_prefix,
        &["q_conv1d", "k_conv1d", "v_conv1d"],
        "qkv_conv.conv",
        "weight",
    )? {
        let fused = weights
            .get(&format!("{attn_prefix}.qkv_conv.conv.weight"))
            .expect("just inserted");
        let shape = mlxcel_core::array_shape(fused);
        let expected = 3 * config.delta_projection_dim() as i32;
        if shape.len() != 3 || shape[0] != expected {
            return Err(format!(
                "{attn_prefix}.qkv_conv.conv.weight: fused conv weight is {shape:?}, expected \
                 [{expected}, kernel, 1] for num_heads * head_dim = {}",
                config.delta_projection_dim()
            ));
        }
    }

    for suffix in ["weight", "scales", "biases"] {
        fuse_axis0(
            weights,
            attn_prefix,
            &["q_proj", "k_proj", "v_proj"],
            "qkv_proj",
            suffix,
        )?;
    }

    // `A_log` is stored `[128]` for 96 heads on the published checkpoint.
    let a_log_key = format!("{attn_prefix}.A_log");
    if let Some(w) = weights.get(&a_log_key) {
        let total: i32 = mlxcel_core::array_shape(w).iter().product();
        let num_heads = config.delta_num_heads() as i32;
        if total > num_heads {
            let flat = mlxcel_core::reshape(w, &[total]);
            let sliced =
                mlxcel_core::contiguous(&mlxcel_core::slice(&flat, &[0], &[num_heads]), false);
            weights.insert(a_log_key, eval_owned(sliced));
        } else if mlxcel_core::array_ndim(w) != 1 {
            let flat = mlxcel_core::reshape(w, &[total]);
            weights.insert(a_log_key, flat);
        }
    }

    let dt_key = format!("{attn_prefix}.dt_bias");
    if let Some(w) = weights.get(&dt_key)
        && mlxcel_core::array_ndim(w) != 1
    {
        let total: i32 = mlxcel_core::array_shape(w).iter().product();
        let flat = mlxcel_core::reshape(w, &[total]);
        weights.insert(dt_key, flat);
    }
    Ok(())
}

/// Reject a per-expert plane set that `stack` would abort on: empty, or with
/// any expert's shape differing from expert 0's.
fn check_uniform_shapes(
    planes: &[UniquePtr<MlxArray>],
    src_prefix: &str,
    src_leaf: &str,
    plane: &str,
) -> Result<(), String> {
    let Some(first) = planes.first() else {
        return Err(format!(
            "{src_prefix}.experts.0.{src_leaf}.{plane}: no expert planes to stack"
        ));
    };
    let expected = mlxcel_core::array_shape(first);
    for (e, w) in planes.iter().enumerate().skip(1) {
        let shape = mlxcel_core::array_shape(w);
        if shape != expected {
            return Err(format!(
                "{src_prefix}.experts.{e}.{src_leaf}.{plane}: shape {shape:?} differs from \
                 expert 0's {expected:?}; every expert plane must have the same shape to stack"
            ));
        }
    }
    Ok(())
}

/// Stack one expert plane. Returns `Ok(false)` when expert 0 carries neither
/// layout (already stacked, or a dense checkpoint under another name).
fn stack_expert_plane(
    weights: &mut WeightMap,
    src_prefix: &str,
    dst_prefix: &str,
    src_leaf: &str,
    dst_leaf: &str,
    num_experts: usize,
) -> Result<bool, String> {
    let expert_key = |e: usize, leaf: &str| format!("{src_prefix}.experts.{e}.{src_leaf}.{leaf}");
    let dst = |plane: &str| format!("{dst_prefix}.switch_mlp.{dst_leaf}.{plane}");

    // compressed-tensors mxfp4: `weight_packed` (uint8) + `weight_scale` (uint8 E8M0).
    if weights.contains_key(&expert_key(0, "weight_packed")) {
        let mut packed = Vec::with_capacity(num_experts);
        let mut scales = Vec::with_capacity(num_experts);
        let mut e = 0;
        while let Some(w) = weights.remove(&expert_key(e, "weight_packed")) {
            let s = weights
                .remove(&expert_key(e, "weight_scale"))
                .ok_or_else(|| {
                    format!(
                        "{}: weight_packed without a weight_scale plane",
                        expert_key(e, "weight_scale")
                    )
                })?;
            if mlxcel_core::array_dtype(&w) != dtype::UINT8
                || mlxcel_core::array_dtype(&s) != dtype::UINT8
            {
                return Err(format!(
                    "{}: mxfp4-pack-quantized experts must ship uint8 weight_packed and uint8 \
                     E8M0 weight_scale planes",
                    expert_key(e, src_leaf)
                ));
            }
            packed.push(w);
            scales.push(s);
            e += 1;
        }
        if e != num_experts {
            return Err(format!(
                "{src_prefix}.experts: checkpoint provides only {e} of the {num_experts} experts \
                 declared by the model config (stacked contiguously from index 0 until the first \
                 gap); refusing to load a truncated MoE layer"
            ));
        }
        // `stack` and `view` are the two MLX calls in this file that abort the
        // process on a bad argument instead of returning: `stack` on planes
        // whose shapes disagree, `view` on a trailing axis that is not a whole
        // number of uint32 words. Both are reachable from checkpoint data, so
        // check them here and report the offending expert by index.
        check_uniform_shapes(&packed, src_prefix, src_leaf, "weight_packed")?;
        check_uniform_shapes(&scales, src_prefix, src_leaf, "weight_scale")?;
        let packed_shape = mlxcel_core::array_shape(&packed[0]);
        let last = *packed_shape.last().ok_or_else(|| {
            format!("{src_prefix}.experts.0.{src_leaf}.weight_packed: expected a shaped plane, got a scalar")
        })?;
        if last <= 0 || last % 4 != 0 {
            return Err(format!(
                "{src_prefix}.experts.0.{src_leaf}.weight_packed: trailing axis {last} is not a \
                 positive multiple of 4, so the uint8 plane is not a whole number of uint32 \
                 mxfp4 words (shape {packed_shape:?})"
            ));
        }
        // [E, out, in / 2] uint8 -> [E, out, in / 8] uint32: the little-endian
        // byte order of the view is the low-nibble-first code order MLX's
        // mxfp4 kernels read, pinned by `mxfp4_repack_matches_scalar_dequant`.
        let stacked = stack_arrays(&packed, 0);
        drop(packed);
        let viewed = eval_owned(mlxcel_core::view(&stacked, dtype::UINT32));
        drop(stacked);
        weights.insert(dst("weight"), viewed);
        let stacked_scales = eval_owned(stack_arrays(&scales, 0));
        drop(scales);
        weights.insert(dst("scales"), stacked_scales);
        return Ok(true);
    }

    // MLX-style per-expert planes (dense, or affine / block-float quantized).
    if weights.contains_key(&expert_key(0, "weight")) {
        for plane in ["weight", "scales", "biases"] {
            if !weights.contains_key(&expert_key(0, plane)) {
                continue;
            }
            let mut sources = Vec::with_capacity(num_experts);
            let mut e = 0;
            while let Some(w) = weights.remove(&expert_key(e, plane)) {
                sources.push(w);
                e += 1;
            }
            if plane == "weight" && e != num_experts {
                return Err(format!(
                    "{src_prefix}.experts: checkpoint provides only {e} of the {num_experts} \
                     experts declared by the model config (stacked contiguously from index 0 \
                     until the first gap); refusing to load a truncated MoE layer"
                ));
            }
            check_uniform_shapes(&sources, src_prefix, src_leaf, plane)?;
            let stacked = eval_owned(stack_arrays(&sources, 0));
            drop(sources);
            weights.insert(dst(plane), stacked);
        }
        return Ok(true);
    }

    Ok(false)
}

fn sanitize_moe_layer(
    weights: &mut WeightMap,
    layer: usize,
    config: &KimiK3TextConfig,
) -> Result<(), String> {
    let src = format!("model.layers.{layer}.block_sparse_moe");
    let dst = format!("model.layers.{layer}.mlp");

    for name in [
        "gate",
        "routed_expert_down_proj",
        "routed_expert_norm",
        "routed_expert_up_proj",
        "shared_experts.gate_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
    ] {
        move_planes(weights, &src, &dst, name);
    }
    if let Some(w) = weights.remove(&format!("{src}.gate.e_score_correction_bias")) {
        weights.insert(format!("{dst}.e_score_correction_bias"), w);
    }

    for (src_leaf, dst_leaf) in [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")] {
        stack_expert_plane(weights, &src, &dst, src_leaf, dst_leaf, config.num_experts)?;
    }

    // Anything still under `experts.` is a plane this loader has no reader
    // for (a compressed-tensors sidecar such as `weight_shape`); it cannot be
    // consumed, so it is dropped rather than left to fail a later lookup.
    let leftovers: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with(&format!("{src}.experts.")))
        .cloned()
        .collect();
    if !leftovers.is_empty() {
        eprintln!(
            "[KimiK3] layer {layer}: dropping {} unread expert tensors (first: {})",
            leftovers.len(),
            leftovers[0]
        );
        for k in leftovers {
            weights.remove(&k);
        }
    }
    Ok(())
}

pub(super) fn sanitize_weights(
    weights: WeightMap,
    config: &KimiK3TextConfig,
) -> Result<WeightMap, String> {
    let num_layers = config.num_hidden_layers;

    // Scope and rename pass.
    let mut weights: WeightMap = weights
        .into_iter()
        .filter_map(|(key, value)| canonical_key(&key, num_layers).map(|k| (k, value)))
        .collect();

    if config.tie_word_embeddings {
        for suffix in ["weight", "scales", "biases"] {
            weights.remove(&format!("lm_head.{suffix}"));
        }
    }

    for layer in 0..num_layers {
        let attn_prefix = format!("model.layers.{layer}.self_attn");
        if config.is_linear_layer(layer) {
            sanitize_kda_layer(&mut weights, &attn_prefix, config)?;
        } else {
            decompose_kv_b_proj(
                &mut weights,
                &attn_prefix,
                layer,
                config.num_attention_heads as i32,
                config.qk_nope_head_dim as i32,
                config.v_head_dim as i32,
                config.kv_lora_rank as i32,
            )?;
        }
        if config.is_moe_layer(layer) {
            sanitize_moe_layer(&mut weights, layer, config)?;
        }
    }

    Ok(weights)
}
