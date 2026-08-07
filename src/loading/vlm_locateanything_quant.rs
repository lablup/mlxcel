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

//! Mixed-precision quantization normalization for the LocateAnything loader.
//!
//! Split out of `vlm_locateanything.rs` because it changes for a different
//! reason than the config parsing and weight-key remapping there: it exists
//! purely to reconcile an `mlx_lm` `mixed_4_8` conversion with mlxcel's fused
//! QKV projection, and would be reusable by any other family that meets the
//! same conversion.

use mlxcel_core::weights::WeightMap;

/// Dequantize `q_proj` / `k_proj` / `v_proj` to dense for every attention layer
/// whose three planes do not share a quantization bit width.
///
/// `mlx_lm`'s `mixed_4_8` quant predicate, which the released LocateAnything
/// conversion uses (the model card explains why: pure 4-bit on the tied
/// `embed_tokens` destroys coordinate-token precision), stores some layers'
/// `v_proj` at 8 bits while `q_proj` / `k_proj` stay at 4. In the released
/// checkpoint that is 18 of the 36 layers.
///
/// `FusedQKVLinear::from_weights_separate` concatenates the three packed planes
/// along axis 0 and infers a single width from `q_proj`. A mixed layer is
/// therefore a hard shape error inside MLX's `concatenate` (4-bit `q` is
/// `[2048, 256]` while 8-bit `v` is `[256, 512]`), and even if the shapes
/// happened to line up, two thirds of the fused tensor would be unpacked at the
/// wrong width. Every non-fused projection (`o_proj`, the MLP, the embedding)
/// reconciles its own width from its own shapes and needs nothing here.
///
/// The fix is to dequantize all three planes of an affected layer and drop
/// their `.scales` / `.biases`, so `FusedQKVLinear` takes its dense branch.
/// Dequantization is exact (it is the stored representation's definition), so
/// this changes no value the model computes with. It is deliberately preferred
/// over requantizing the narrow planes up to the wider width: MLX's affine
/// quantizer snaps the group scale onto the larger-magnitude edge rather than
/// using a plain `(max - min) / (2^bits - 1)`, so a 4-bit group does not land
/// exactly on the 8-bit grid and a round trip would perturb the weights by up
/// to half an 8-bit step.
///
/// The cost is memory and bandwidth on the affected layers: for the released
/// 3B checkpoint the QKV planes of 18 layers become bf16, about 190 MB more
/// than their packed form. Removing it needs a `FusedQKVLinear` that can hold
/// per-projection widths, which is a shared-layer change and out of scope here.
/// All three planes are converted together, and never only the odd one out,
/// because `from_weights_separate` decides quantized-vs-dense from `q_proj`
/// alone.
///
/// Returns the number of layers that were converted.
pub(super) fn densify_mixed_precision_qkv(
    weights: &mut WeightMap,
    group_size: i32,
    declared_bits: i32,
    mode: &str,
) -> Result<usize, String> {
    const PROJECTIONS: [&str; 3] = ["q_proj", "k_proj", "v_proj"];

    let mut prefixes: Vec<String> = weights
        .keys()
        .filter_map(|k| k.strip_suffix(".q_proj.scales"))
        .filter(|p| p.ends_with(".self_attn"))
        .map(|p| p.to_string())
        .collect();
    prefixes.sort();

    let mut converted = 0usize;
    for prefix in prefixes {
        let mut layouts = Vec::with_capacity(PROJECTIONS.len());
        for proj in PROJECTIONS {
            let w = weights.get(&format!("{prefix}.{proj}.weight"));
            let s = weights.get(&format!("{prefix}.{proj}.scales"));
            let (Some(w), Some(s)) = (w, s) else {
                // A partially quantized triple is not something to rewrite;
                // leave it for the backbone loader to report.
                layouts.clear();
                break;
            };
            let layout = mlxcel_core::layers::reconcile_quantization_layout(
                &mlxcel_core::array_shape(w),
                &mlxcel_core::array_shape(s),
                group_size,
                declared_bits,
                mode,
            )
            .map_err(|e| format!("{prefix}.{proj}: {e}"))?;
            layouts.push(layout);
        }
        if layouts.len() != PROJECTIONS.len() {
            continue;
        }
        if layouts.iter().all(|l| l.bits == layouts[0].bits)
            && layouts
                .iter()
                .all(|l| l.group_size == layouts[0].group_size)
        {
            continue;
        }

        for (proj, layout) in PROJECTIONS.iter().zip(layouts.iter()) {
            dequantize_plane_in_place(
                weights,
                &format!("{prefix}.{proj}"),
                layout.group_size,
                layout.bits,
                mode,
            )?;
        }
        converted += 1;
    }

    Ok(converted)
}

/// Replace `{prefix}.weight` with its dequantized dense form and drop the
/// `.scales` / `.biases` planes that described the packing.
///
/// The true linear `.bias` tensor (which Qwen2 attention carries on q/k/v) is a
/// different key and is deliberately left alone.
fn dequantize_plane_in_place(
    weights: &mut WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Result<(), String> {
    let dense = {
        let w = weights
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
        let s = weights
            .get(&format!("{prefix}.scales"))
            .ok_or_else(|| format!("Scales not found: {prefix}.scales"))?;
        let b_ptr = weights
            .get(&format!("{prefix}.biases"))
            .and_then(|b| b.as_ref())
            .map(|r| r as *const mlxcel_core::MlxArray)
            .unwrap_or(std::ptr::null());
        // SAFETY: `w` and `s` are borrowed from live map entries for the
        // duration of the call, and `b_ptr` is either null or borrowed from a
        // live entry in the same map.
        let dense = unsafe { mlxcel_core::dequantize(w, s, b_ptr, group_size, bits, mode) };
        mlxcel_core::eval(&dense);
        dense
    };

    weights.insert(format!("{prefix}.weight"), dense);
    weights.remove(&format!("{prefix}.scales"));
    weights.remove(&format!("{prefix}.biases"));
    Ok(())
}

#[cfg(test)]
#[path = "vlm_locateanything_quant_tests.rs"]
mod tests;
