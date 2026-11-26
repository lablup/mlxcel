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

//! The NVFP4 exports of Raon-1 mini (issue #3437).
//!
//! - `KRAFTON/Raon-1-mini-mlx` (vendor MLX layout): an MLX `quantization`
//!   block, NVFP4 g16 everywhere except an MXFP8 g32 `lm_head`. Routed
//!   experts carry a per-expert `global_scale` flagged `"global_scale": true`,
//!   which `load_text_weights` resolves for `gather_qmm`. The linears listed
//!   in `raon_global_scaled_linears` carry a per-tensor `global_scale` that
//!   multiplies the output, `bf16(f32(y) * g)`, as `UnifiedLinear` applies a
//!   ModelOpt `weight_scale_2`. The vendor runtime switches
//!   (`raon_merged_gate_up`, `raon_fused_mhc_decode`, `model_file`) and the
//!   optional `head_stage1_*.safetensors` / `dflash2/` files are not read.
//! - `KRAFTON/Raon-1-mini-nvfp4` (NVIDIA ModelOpt `MIXED_PRECISION`): the
//!   shared NVFP4 repack turns every triplet into an NVFP4 plane with an
//!   output multiplier and the MXFP8 `lm_head` into MLX's `mxfp8` plane;
//!   here the per-expert planes are stacked into `switch_mlp`.
//!
//! The layout of each quantized linear comes from its tensor shapes, so one
//! loader serves the original checkpoint (bf16 linears, MXFP8 experts) and
//! both exports.

use super::raon1_mini_config::Raon1MiniConfig;
use crate::models::modelopt_mixed_precision as mixed;
use crate::models::switch_layers::{drain_numbered_expert_planes, infer_block_float_layout};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;

const EXPERT_PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Refuse a quantization this loader would run wrong. Accepts the original
/// checkpoint's `mxfp8` `quantization_config`, the vendor MLX export's
/// `quantization` block, and a ModelOpt mixed-precision declaration whose
/// layers are all NVFP4, W4A16_NVFP4 or MXFP8.
pub(super) fn check_export(config: &Value) -> Result<(), String> {
    let fail = |msg: String| Err(format!("raon1_mini: {msg}"));
    let mlx = config.get("quantization");
    if let Some(q) = mlx {
        check_mlx_quantization(q).or_else(fail)?;
    }
    let Some(qc) = config.get("quantization_config") else {
        return Ok(());
    };
    if mixed::is_mixed_precision_declaration(qc) {
        mixed::mixed_precision_mxfp8_layers(qc).map_err(|e| format!("raon1_mini: {e}"))?;
        if qc.get("kv_cache_scheme").is_some_and(|s| !s.is_null()) {
            return fail(format!(
                "ModelOpt kv_cache_scheme {:?} is not supported; the cache stays in the \
                 activation dtype",
                qc.get("kv_cache_scheme")
            ));
        }
        return Ok(());
    }
    // mlx-lm writes its `quantization` block a second time as
    // `quantization_config`.
    if mlx.is_some_and(|q| q == qc) {
        return Ok(());
    }
    let method = qc.get("quant_method").and_then(Value::as_str).unwrap_or("");
    if method != "mxfp8" {
        return fail(format!(
            "quantization_config quant_method {method:?} is not supported; the original \
             checkpoint (native mxfp8 experts), KRAFTON/Raon-1-mini-mlx and \
             KRAFTON/Raon-1-mini-nvfp4 (modelopt_fp4 MIXED_PRECISION) load"
        ));
    }
    if qc.get("dequantize").and_then(Value::as_bool) == Some(true) {
        return fail("quantization_config dequantize true is not supported".into());
    }
    let block = qc.get("weight_block_size").and_then(Value::as_array);
    let block_ok =
        block.is_none_or(|b| b.len() == 2 && b[0].as_u64() == Some(1) && b[1].as_u64() == Some(32));
    if !block_ok {
        return fail(format!(
            "mxfp8 weight_block_size {:?} is not supported; MLX's mxfp8 is [1, 32]",
            qc.get("weight_block_size")
        ));
    }
    Ok(())
}

/// An MLX `quantization` block: an NVFP4 g16 default with per-module
/// overrides that are NVFP4 g16 (optionally `global_scale`) or MXFP8 g32.
fn check_mlx_quantization(q: &Value) -> Result<(), String> {
    let Some(entries) = q.as_object() else {
        return Err("`quantization` must be an object".into());
    };
    fn layout(v: &Value) -> (&str, Option<i64>, Option<i64>) {
        (
            v.get("mode").and_then(Value::as_str).unwrap_or("affine"),
            v.get("group_size").and_then(Value::as_i64),
            v.get("bits").and_then(Value::as_i64),
        )
    }
    if layout(q) != ("nvfp4", Some(16), Some(4)) {
        return Err(format!(
            "MLX quantization {:?} is not supported; the vendor export is nvfp4 with group_size \
             16 and bits 4 (KRAFTON/Raon-1-mini-mlx)",
            layout(q)
        ));
    }
    for (path, entry) in entries.iter().filter(|(_, v)| v.is_object()) {
        match layout(entry) {
            ("nvfp4", Some(16), Some(4)) | ("mxfp8", Some(32), Some(8)) => {}
            other => {
                return Err(format!(
                    "MLX quantization entry {path} is {other:?}; only nvfp4 (16, 4) and mxfp8 \
                     (32, 8) layers load"
                ));
            }
        }
        if entry.get("global_scale").is_some_and(|g| !g.is_boolean()) {
            return Err(format!(
                "MLX quantization entry {path}: global_scale must be a boolean"
            ));
        }
    }
    Ok(())
}

impl Raon1MiniConfig {
    /// `(group_size, bits)` the RDA and MLA linears are declared with: the
    /// MLX block's default, 16 / 4 for a ModelOpt export, else the unused
    /// affine default (the original checkpoint's linears are bf16).
    pub(crate) fn dense_quant(&self) -> (i32, i32) {
        if let Some(q) = self.extra.get("quantization") {
            let field = |k: &str| {
                q.get(k)
                    .and_then(Value::as_i64)
                    .and_then(|v| i32::try_from(v).ok())
            };
            if let (Some(group_size), Some(bits)) = (field("group_size"), field("bits")) {
                return (group_size, bits);
            }
        }
        if self
            .extra
            .get("quantization_config")
            .is_some_and(mixed::is_mixed_precision_declaration)
        {
            return (16, 4);
        }
        (64, 4)
    }
}

/// A dense linear whose block-float layout (NVFP4 g16, MXFP8 g32, MXFP4 g32)
/// is read from its `uint8` scales, so an NVFP4 projection and the MXFP8
/// `lm_head` (whose packed widths an NVFP4 or MXFP4 reading would also fit)
/// load without a declared layout. Anything else takes `UnifiedLinear`'s own
/// detection; a bf16 linear loads unquantized.
pub(crate) fn linear(weights: &WeightMap, prefix: &str) -> Result<UnifiedLinear, String> {
    let layout = match (
        weights.get(&format!("{prefix}.weight")),
        weights.get(&format!("{prefix}.scales")),
    ) {
        (Some(w), Some(s)) if !weights.contains_key(&format!("{prefix}.biases")) => {
            infer_block_float_layout(w, s)
        }
        _ => None,
    };
    match layout {
        Some((mode, group_size)) => {
            let bits = if mode == "mxfp8" { 8 } else { 4 };
            UnifiedLinear::from_weights_with_mode(weights, prefix, group_size, bits, mode)
        }
        None => UnifiedLinear::from_weights(weights, prefix, 64, 4),
    }
}

/// The export-specific part of sanitize. Stacks the ModelOpt export's
/// per-expert NVFP4 planes (already repacked to `weight` / `scales` /
/// `global_scale`) into `{moe}.switch_mlp.{proj}`, and checks the vendor
/// export's per-tensor scales against `raon_global_scaled_linears`. A no-op
/// on the original checkpoint, whose MXFP8 experts carry `weight_scale_inv`.
pub(super) fn sanitize_export(
    weights: &mut WeightMap,
    config: &Raon1MiniConfig,
) -> Result<(), String> {
    for layer in (0..config.num_hidden_layers).filter(|&l| config.is_moe_layer(l)) {
        let moe = format!("model.layers.{layer}.moe");
        if weights.contains_key(&format!("{moe}.switch_mlp.gate_up.weight")) {
            return Err(format!(
                "{moe}.switch_mlp.gate_up.weight: merged gate_up expert planes are not \
                 supported; the published export keeps gate_proj and up_proj separate"
            ));
        }
        for proj in EXPERT_PROJS {
            if weights.contains_key(&format!("{moe}.experts.0.{proj}.scales")) {
                drain_numbered_expert_planes(
                    weights,
                    &format!("{moe}.experts"),
                    proj,
                    &format!("{moe}.switch_mlp.{proj}"),
                )?;
            }
        }
    }
    check_global_scaled_linears(weights, config)
}

/// Every per-tensor `global_scale` outside the expert stacks is one of the
/// config's `raon_global_scaled_linears`, and every listed linear has one: a
/// scale the vendor does not apply, or a listed one missing, would change the
/// linear's output silently.
fn check_global_scaled_linears(
    weights: &WeightMap,
    config: &Raon1MiniConfig,
) -> Result<(), String> {
    let Some(listed) = config.extra.get("raon_global_scaled_linears") else {
        return Ok(());
    };
    let listed: Vec<&str> = listed
        .as_array()
        .ok_or("raon1_mini: raon_global_scaled_linears must be a list of module paths")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or("raon1_mini: raon_global_scaled_linears must be a list of module paths")
        })
        .collect::<Result<_, _>>()?;
    if let Some(stray) = weights.keys().find(|k| {
        k.strip_suffix(".global_scale")
            .is_some_and(|path| !path.contains(".switch_mlp.") && !listed.contains(&path))
    }) {
        return Err(format!(
            "{stray}: a per-tensor global_scale on a linear raon_global_scaled_linears does not \
             list"
        ));
    }
    if let Some(missing) = listed.iter().find(|path| {
        weights.contains_key(&format!("{path}.weight"))
            && !weights.contains_key(&format!("{path}.global_scale"))
    }) {
        return Err(format!(
            "{missing}.global_scale is missing; raon_global_scaled_linears lists {missing}"
        ));
    }
    Ok(())
}
