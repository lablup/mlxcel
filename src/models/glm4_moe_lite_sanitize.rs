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

//! Weight-map sanitization for GLM4 MoE Lite.
//!
//! Split out of `glm4_moe_lite.rs` rather than appended to it, the same way
//! `youtu_vl_lm_sanitize.rs` was split out of `youtu_vl_lm.rs`: the runtime
//! module is already well past the 500-line target and this keeps the new code
//! from pushing it further. [`sanitize_weights`] is re-exported from
//! `glm4_moe_lite` so call sites name it the way the other MLA families do.

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;

use super::ModelArgs;

/// Decompose `kv_b_proj` into the per-head `embed_q` / `unembed_out` pair that
/// `MlaAttention::from_weights` loads.
///
/// Every public `glm4_moe_lite` checkpoint stores the MLA up-projection as a
/// single `self_attn.kv_b_proj.weight` and ships no `embed_q` tensor at all, so
/// without this step the family cannot load the canonical layout of its own
/// architecture: `MultiLinear::from_weights` fails with `Weight not found:
/// model.layers.0.self_attn.embed_q.weight` (issue #1029). The other five MLA
/// families in the tree (DeepSeek V3, DeepSeek V3.2, Kimi Linear, LongCat Flash
/// NGram, Youtu-VL) all carry the same decomposition; this one is adapted from
/// `youtu_vl_lm_sanitize.rs`, the only one that cross-checks the tensor shape
/// against the config before reshaping.
///
/// A quantized `kv_b_proj` is dequantized before the split, because the split
/// runs per head along the row axis and slicing in packed space would leave
/// each half carrying group scales and biases that describe the whole row.
///
/// Returns `Err` rather than panicking or aborting when a `kv_b_proj` cannot be
/// decomposed: scales with no biases, a solved quantization pair no packing can
/// describe, or a tensor whose shape disagrees with `config.json`.
///
/// Used by: [`super::Glm4MoeLiteModel::load`].
pub fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> Result<WeightMap, String> {
    let num_heads = args.num_attention_heads as i32;
    let head_dim = (args.qk_nope_head_dim + args.v_head_dim) as i32;
    let qk_nope_head_dim = args.qk_nope_head_dim as i32;
    let kv_lora_rank = args.kv_lora_rank as i32;

    for layer_idx in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}.self_attn");
        let kv_b_key = format!("{prefix}.kv_b_proj.weight");
        let embed_q_key = format!("{prefix}.embed_q.weight");

        // A checkpoint that already ships the decomposed pair is left exactly as
        // it is, matching the sibling sanitizers. Re-deriving the pair would at
        // best duplicate work and at worst replace a quantized `embed_q` (which
        // `MultiLinear` still loads through `QuantizedMultiLinear`) with a dense
        // one built from a `kv_b_proj` the exporter may have left behind stale.
        if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
            continue;
        }

        let scales_key = format!("{prefix}.kv_b_proj.scales");
        let is_quantized = weights.contains_key(&scales_key);

        let w = weights.remove(&kv_b_key).unwrap();

        let w_full = if is_quantized {
            let s = weights.remove(&scales_key).unwrap();
            // `is_quantized` gates on `.scales` alone, and the block-float modes
            // (mxfp4 / nvfp4 / mxfp8) ship scales with no zero points, so a
            // block-float export satisfies that gate and arrives here carrying
            // no `.biases` plane. Taking it with `.unwrap()` would turn that
            // into a panic during sanitization, which in the server takes the
            // process down rather than rejecting one model load (issue #1026
            // aligned the other five sanitizers on this wording). `dequantize`
            // below is hardcoded `"affine"` and so could not decompose such a
            // plane in any case; what this buys is a load error naming the key
            // that is missing.
            let b_key = format!("{prefix}.kv_b_proj.biases");
            let b = weights.remove(&b_key).ok_or_else(|| {
                format!(
                    "layer {layer_idx}: kv_b_proj has scales but no biases at key `{b_key}`; \
                     the checkpoint may be corrupted or only partially converted"
                )
            })?;

            // Solve the packed pair from the shapes and bound it before it
            // reaches `dequantize`. The shared helper also checks each divisor
            // before dividing: `kv_lora_rank` is a config field and the scales
            // axis is checkpoint data, so the naive form panics on a zero
            // divisor and overflows i32 on a large packed axis, both before the
            // bound could fire (issue #958).
            let (inferred_gs, inferred_bits) = mlxcel_core::layers::infer_mla_quantization_params(
                &mlxcel_core::array_shape(&w),
                &mlxcel_core::array_shape(&s),
                kv_lora_rank,
                &format!("{prefix}.kv_b_proj"),
            )?;

            // SAFETY: `w`, `s` and `b` are live UniquePtr-owned arrays taken out
            // of the weight map above and are not dropped until this call
            // returns.
            unsafe {
                mlxcel_core::dequantize(
                    &w,
                    &s,
                    &*b as *const _,
                    inferred_gs,
                    inferred_bits,
                    "affine",
                )
            }
        } else {
            mlxcel_core::copy(&w)
        };

        // Cross-check the tensor against the config before reshaping. A
        // mismatch means `config.json` and the stored tensor disagree, and the
        // reshape below is the point where that stops being recoverable: MLX
        // reports a bad reshape by throwing, and the throw crosses the cxx
        // bridge as `UniquePtr<MlxArray>` rather than `Result`, so it aborts the
        // process during weight sanitization instead of failing the load. Of
        // the five sibling sanitizers only Youtu-VL carries this check.
        let w_shape = mlxcel_core::array_shape(&w_full);
        let expected_rows = num_heads * head_dim;
        if w_shape.len() != 2 || w_shape[0] != expected_rows || w_shape[1] != kv_lora_rank {
            return Err(format!(
                "layer {layer_idx}: kv_b_proj shape mismatch: got {w_shape:?}, expected \
                 [{expected_rows}, {kv_lora_rank}] (num_heads={num_heads}, head_dim={head_dim}, \
                 kv_lora_rank={kv_lora_rank})"
            ));
        }

        // [num_heads * head_dim, kv_lora_rank] -> [num_heads, head_dim, kv_lora_rank]
        let w_3d = mlxcel_core::reshape(&w_full, &[num_heads, head_dim, -1]);

        // wk = w[:, :qk_nope_head_dim, :]  (the nope half)
        // wv = w[:, qk_nope_head_dim:, :]  (the v half)
        let wk = slice_axis(&w_3d, 1, 0, qk_nope_head_dim);
        let wv = slice_axis(&w_3d, 1, qk_nope_head_dim, -1);

        // embed_q stores the swapped-axes form: [num_heads, kv_lora_rank, qk_nope_head_dim]
        let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);

        // Copy both halves so `MultiLinear`'s matmul sees well-formed strides
        // regardless of the backend, rather than a view into the parent tensor.
        let wk = mlxcel_core::copy(&wk);
        let wv = mlxcel_core::copy(&wv);

        weights.insert(format!("{prefix}.embed_q.weight"), wk);
        weights.insert(format!("{prefix}.unembed_out.weight"), wv);
    }

    Ok(weights)
}

#[cfg(test)]
#[path = "glm4_moe_lite_sanitize_tests.rs"]
mod tests;
