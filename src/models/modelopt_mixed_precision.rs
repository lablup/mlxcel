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

//! NVIDIA ModelOpt `MIXED_PRECISION` exports (issue #3437).
//!
//! A mixed-precision export declares its format per layer instead of through
//! `config_groups`:
//!
//! ```json
//! "quantization_config": {
//!   "quant_method": "modelopt_fp4", "quant_algo": "MIXED_PRECISION", "group_size": 16,
//!   "quantized_layers": {
//!     "model.layers.1.moe.experts.0.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
//!     "model.layers.0.self_attn.q_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
//!     "lm_head": {"quant_algo": "MXFP8", "group_size": 32}
//!   }
//! }
//! ```
//!
//! NVFP4 and W4A16_NVFP4 layers are the usual `{weight, weight_scale,
//! weight_scale_2}` triplets the NVFP4 repack in `sanitize.rs` already
//! handles. An MXFP8 layer stores E4M3 codes (`weight`, `[out, in]` bytes) and
//! one E8M0 scale per 32 inputs (`weight_scale`, `[out, in / 32]` bytes), which
//! is MLX's native `mxfp8` layout once the codes are viewed as uint32. Any
//! other per-layer algorithm is a named load error.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{array_dtype, array_shape, dtype};
use serde_json::Value;

/// ModelOpt NVFP4 block size.
const NVFP4_GROUP: i64 = 16;
/// MLX `mxfp8` block size.
const MXFP8_GROUP: i64 = 32;

/// Whether `qc` declares a ModelOpt mixed-precision export: `quant_method`
/// `modelopt` or `modelopt_fp4`, `quant_algo` `MIXED_PRECISION`, and a
/// `quantized_layers` object. Says nothing about whether its layers load;
/// [`mixed_precision_mxfp8_layers`] decides that.
pub(crate) fn is_mixed_precision_declaration(qc: &Value) -> bool {
    let text = |key: &str| {
        qc.get(key)
            .and_then(Value::as_str)
            .map(|v| v.trim().to_ascii_lowercase())
    };
    matches!(
        text("quant_method").as_deref(),
        Some("modelopt" | "modelopt_fp4")
    ) && text("quant_algo").as_deref() == Some("mixed_precision")
        && qc.get("quantized_layers").is_some_and(Value::is_object)
}

/// The MXFP8 layer paths of a mixed-precision declaration, after checking
/// that every layer is one mlxcel loads: NVFP4 or W4A16_NVFP4 at group size
/// 16, or MXFP8 at group size 32, with at least one NVFP4 layer. A missing
/// `group_size` takes the format's own block size. Any other algorithm or
/// group size is an error naming the layer.
///
/// Used by: [`crate::models::gemma4::is_modelopt_nvfp4`] (detection),
/// [`repack_mixed_precision_mxfp8_layers`], and the Raon-1 mini config check
/// (`raon1_mini_quant.rs`)
pub(crate) fn mixed_precision_mxfp8_layers(qc: &Value) -> Result<Vec<String>, String> {
    let Some(layers) = qc.get("quantized_layers").and_then(Value::as_object) else {
        return Err("ModelOpt MIXED_PRECISION export has no quantized_layers object".into());
    };
    let mut mxfp8 = Vec::new();
    let mut nvfp4 = 0usize;
    for (path, entry) in layers {
        let algo = entry
            .get("quant_algo")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_uppercase();
        let group = entry.get("group_size").and_then(Value::as_i64);
        let (want, is_mxfp8) = match algo.as_str() {
            "NVFP4" | "W4A16_NVFP4" => (NVFP4_GROUP, false),
            "MXFP8" => (MXFP8_GROUP, true),
            other => {
                return Err(format!(
                    "ModelOpt MIXED_PRECISION layer {path} uses quant_algo {other:?}; only \
                     NVFP4, W4A16_NVFP4 and MXFP8 layers load"
                ));
            }
        };
        if group.is_some_and(|g| g != want) {
            return Err(format!(
                "ModelOpt MIXED_PRECISION layer {path} ({algo}) has group_size {group:?}; \
                 {algo} needs {want}"
            ));
        }
        if is_mxfp8 {
            mxfp8.push(path.clone());
        } else {
            nvfp4 += 1;
        }
    }
    if nvfp4 == 0 {
        return Err("ModelOpt MIXED_PRECISION export lists no NVFP4 layer".into());
    }
    mxfp8.sort();
    Ok(mxfp8)
}

/// The mixed-precision `quantization_config` of `config` (top level or
/// `text_config`), when it declares one.
fn mixed_precision_declaration(config: &Value) -> Option<&Value> {
    std::iter::once(config)
        .chain(config.get("text_config"))
        .filter_map(|root| root.get("quantization_config"))
        .find(|qc| is_mixed_precision_declaration(qc))
}

/// Families whose loaders read a linear's block-float layout from its tensor
/// shapes, so a repacked MXFP8 plane loads as `mxfp8` g32. Every other family
/// threads its config's NVFP4 16 / 4 into `UnifiedLinear::from_weights`, which
/// reconciles an MXFP8 plane's widths to an MXFP4 reading at group size 64
/// that MLX cannot run; its MXFP8 layers stay a named load error.
const MXFP8_LAYER_FAMILIES: [&str; 1] = ["raon1_mini"];

/// Whether `config` (top level or `text_config`) names a family in
/// [`MXFP8_LAYER_FAMILIES`].
fn loads_mxfp8_layers(config: &Value) -> bool {
    std::iter::once(config)
        .chain(config.get("text_config"))
        .filter_map(|root| root.get("model_type").and_then(Value::as_str))
        .any(|family| MXFP8_LAYER_FAMILIES.contains(&family))
}

/// Rewrite the MXFP8 layers of a mixed-precision export into MLX's `mxfp8`
/// plane, in place and byte for byte: `{p}.weight` (uint8 E4M3 `[out, in]`)
/// is viewed as uint32 `[out, in / 4]`, `{p}.weight_scale` (uint8 E8M0
/// `[out, in / 32]`) becomes `{p}.scales`, and `{p}.input_scale` is dropped.
/// A no-op for a config that declares no mixed-precision export; a listed
/// layer whose `weight` is not in the map is skipped (a filtered load). An
/// MXFP8 layer of a family outside [`MXFP8_LAYER_FAMILIES`] fails by name.
///
/// Used by: `sanitize_modelopt_nvfp4_weights` (`src/models/sanitize.rs`),
/// which every text family's ModelOpt load goes through; Raon1Mini's
/// `lm_head` is its first MXFP8 layer
pub(crate) fn repack_mixed_precision_mxfp8_layers(
    weights: &mut WeightMap,
    config: Option<&Value>,
) -> Result<(), String> {
    let Some((config, qc)) = config.and_then(|c| mixed_precision_declaration(c).map(|qc| (c, qc)))
    else {
        return Ok(());
    };
    let family_loads_mxfp8 = loads_mxfp8_layers(config);
    for path in mixed_precision_mxfp8_layers(qc)? {
        let w_key = format!("{path}.weight");
        let s_key = format!("{path}.weight_scale");
        let Some(w) = weights.get(&w_key) else {
            continue;
        };
        if !family_loads_mxfp8 {
            return Err(format!(
                "ModelOpt MIXED_PRECISION layer {path} is MXFP8, which this model family does \
                 not load; only {MXFP8_LAYER_FAMILIES:?} read MXFP8 layers in a mixed-precision \
                 export"
            ));
        }
        let s = weights.get(&s_key).ok_or_else(|| {
            format!("{s_key} is missing; the MXFP8 layer {path} needs its E8M0 block scales")
        })?;
        for (key, arr, what) in [(&w_key, w, "E4M3 codes"), (&s_key, s, "E8M0 scales")] {
            let got = array_dtype(arr);
            if got != dtype::UINT8 {
                return Err(format!(
                    "{key}: expected uint8 {what} for the MXFP8 layer, got dtype {got}"
                ));
            }
        }
        let (ws, ss) = (array_shape(w), array_shape(s));
        if ws.len() != 2
            || ss.len() != 2
            || ws[0] != ss[0]
            || ws[1] % 32 != 0
            || ss[1] != ws[1] / 32
        {
            return Err(format!(
                "{w_key} {ws:?} / {s_key} {ss:?}: an MXFP8 layer is [out, in] codes with \
                 [out, in / 32] scales"
            ));
        }
        let packed = mlxcel_core::view(&mlxcel_core::contiguous(w, false), dtype::UINT32);
        let scales = weights.remove(&s_key).expect("checked above");
        weights.insert(w_key, packed);
        weights.insert(format!("{path}.scales"), scales);
        weights.remove(&format!("{path}.input_scale"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "modelopt_mixed_precision_tests.rs"]
mod tests;
