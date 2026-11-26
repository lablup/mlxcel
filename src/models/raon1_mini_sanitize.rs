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

//! Weight sanitize for the original `KRAFTON/Raon-1-mini` checkpoint.
//!
//! - Layers `num_hidden_layers..` (next-token-prediction heads) are dropped.
//! - RDA: `q_conv1d`, `k_conv1d`, `v_conv1d` are concatenated in that order
//!   along the channel axis into `qkv_conv1d.weight` (`[3P, K, 1]`). Each
//!   source may be `[C, K]` (the checkpoint) or `[C, 1, K]` (PyTorch conv
//!   layout); `[C, K, 1]` is accepted as already converted.
//! - MLA: `kv_b_proj.weight` is split into `embed_q` / `unembed_out`
//!   (`kimi_linear::decompose_kv_b_proj`).
//! - Routed experts: each `moe.experts.E.{gate,up,down}_proj` pair of uint8
//!   E4M3 `weight` and uint8 E8M0 `weight_scale_inv` is already MLX's native
//!   `mxfp8` layout. The codes are stacked and viewed as uint32 (no byte
//!   changes), the scales renamed `scales`, under `moe.switch_mlp.{proj}`.
//!   A weight or scale of any other dtype is a load error.
//!
//! Every step is a no-op on its own output, so the pass is idempotent.

use super::raon1_mini_config::Raon1MiniConfig;
use crate::models::kimi_linear::decompose_kv_b_proj;
use mlxcel_core::utils::stack_arrays;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const CONV_LEAVES: [&str; 3] = ["q_conv1d", "k_conv1d", "v_conv1d"];
const EXPERT_PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Rewrite the checkpoint into the layout `Raon1MiniModel::from_weights` reads.
pub fn sanitize_weights(
    mut weights: WeightMap,
    config: &Raon1MiniConfig,
) -> Result<WeightMap, String> {
    drop_extra_layers(&mut weights, config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let attn = format!("model.layers.{layer}.self_attn");
        if config.is_rda_layer(layer) {
            fuse_rda_convs(
                &mut weights,
                &attn,
                config.rda_projection_dim(),
                config.rda_config.conv_kernel_size,
            )?;
        } else {
            decompose_kv_b_proj(
                &mut weights,
                &attn,
                layer,
                config.num_attention_heads as i32,
                config.qk_nope_head_dim as i32,
                config.v_head_dim as i32,
                config.kv_lora_rank as i32,
            )?;
        }
        if config.is_moe_layer(layer) {
            stack_mxfp8_experts(
                &mut weights,
                &format!("model.layers.{layer}.moe"),
                config.num_experts,
            )?;
        }
    }
    Ok(weights)
}

fn drop_extra_layers(weights: &mut WeightMap, num_layers: usize) {
    let extra: Vec<String> = weights
        .keys()
        .filter(|key| {
            key.strip_prefix("model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
                .is_some_and(|idx| idx >= num_layers)
        })
        .cloned()
        .collect();
    for key in extra {
        weights.remove(&key);
    }
}

/// One conv weight as `[C, K, 1]`, from `[C, K]`, `[C, 1, K]` or `[C, K, 1]`.
pub fn conv_to_mlx_layout(
    w: &MlxArray,
    kernel: usize,
    key: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(w);
    let k = kernel as i32;
    match shape.as_slice() {
        [_, kk] if *kk == k => Ok(mlxcel_core::expand_dims(w, -1)),
        [_, 1, kk] if *kk == k && k != 1 => Ok(mlxcel_core::swap_axes(w, -1, -2)),
        [_, kk, 1] if *kk == k => Ok(mlxcel_core::copy(w)),
        _ => Err(format!(
            "{key}: shape {shape:?} is not a depthwise conv of kernel {kernel} ([C, K], [C, 1, K] or [C, K, 1])"
        )),
    }
}

fn fuse_rda_convs(
    weights: &mut WeightMap,
    attn: &str,
    channels: usize,
    kernel: usize,
) -> Result<(), String> {
    let keys: Vec<String> = CONV_LEAVES
        .iter()
        .map(|leaf| format!("{attn}.{leaf}.weight"))
        .collect();
    if !keys.iter().any(|k| weights.contains_key(k)) {
        return Ok(());
    }
    let mut parts = Vec::with_capacity(3);
    for key in &keys {
        let w = weights.remove(key).ok_or_else(|| {
            format!("Missing weight: {key} (the RDA convs come as a q, k, v triple)")
        })?;
        let w = conv_to_mlx_layout(&w, kernel, key)?;
        let rows = mlxcel_core::array_shape(&w)[0];
        if rows as usize != channels {
            return Err(format!(
                "{key}: {rows} channels, expected rda num_heads * head_dim = {channels}"
            ));
        }
        parts.push(w);
    }
    let fused = mlxcel_core::concatenate_many(&[&*parts[0], &*parts[1], &*parts[2]], 0);
    weights.insert(format!("{attn}.qkv_conv1d.weight"), fused);
    Ok(())
}

/// Stack `{moe}.experts.E.{proj}` MXFP8 pairs into `{moe}.switch_mlp.{proj}`.
pub fn stack_mxfp8_experts(
    weights: &mut WeightMap,
    moe: &str,
    num_experts: usize,
) -> Result<(), String> {
    for proj in EXPERT_PROJS {
        let first = format!("{moe}.experts.0.{proj}.weight");
        if !weights.contains_key(&first) {
            continue;
        }
        let mut codes = Vec::with_capacity(num_experts);
        let mut scales = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w_key = format!("{moe}.experts.{e}.{proj}.weight");
            let s_key = format!("{moe}.experts.{e}.{proj}.weight_scale_inv");
            let w = weights
                .remove(&w_key)
                .ok_or_else(|| format!("Missing weight: {w_key}"))?;
            let s = weights
                .remove(&s_key)
                .ok_or_else(|| format!("Missing weight: {s_key}"))?;
            let (w, s) = check_mxfp8_pair(&w_key, w, &s_key, s)?;
            codes.push(w);
            scales.push(s);
        }
        if weights.contains_key(&format!("{moe}.experts.{num_experts}.{proj}.weight")) {
            return Err(format!(
                "{moe}.experts.{num_experts}.{proj}.weight exists, but num_experts is {num_experts}"
            ));
        }
        let stacked = stack_arrays(&codes, 0);
        let packed = mlxcel_core::view(&mlxcel_core::contiguous(&stacked, false), dtype::UINT32);
        weights.insert(format!("{moe}.switch_mlp.{proj}.weight"), packed);
        weights.insert(
            format!("{moe}.switch_mlp.{proj}.scales"),
            stack_arrays(&scales, 0),
        );
    }
    Ok(())
}

fn check_mxfp8_pair(
    w_key: &str,
    w: UniquePtr<MlxArray>,
    s_key: &str,
    s: UniquePtr<MlxArray>,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
    for (key, arr, what) in [(w_key, &w, "E4M3 codes"), (s_key, &s, "E8M0 scales")] {
        let got = mlxcel_core::array_dtype(arr);
        if got != dtype::UINT8 {
            return Err(format!(
                "{key}: expected uint8 {what} (the checkpoint's native mxfp8), got dtype {got}; \
                 dequantized or requantized expert planes are not supported"
            ));
        }
    }
    let ws = mlxcel_core::array_shape(&w);
    let ss = mlxcel_core::array_shape(&s);
    if ws.len() != 2 || ss.len() != 2 || ws[0] != ss[0] || ws[1] % 32 != 0 || ss[1] != ws[1] / 32 {
        return Err(format!(
            "{w_key} {ws:?} / {s_key} {ss:?}: expected [out, in] codes with [out, in / 32] scales"
        ));
    }
    Ok((w, s))
}
