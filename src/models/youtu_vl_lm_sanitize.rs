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

//! Weight-map sanitization for the Youtu-VL language model.
//!
//! Lifted out of `youtu_vl_lm.rs` to keep the runtime module under the
//! 500-line soft target while preserving the same public API. The exported
//! [`sanitize_text_weights`] function continues to be re-exported from
//! `youtu_vl_lm` so existing call sites (loader, tests) do not change.

use mlxcel_core::weights::WeightMap;

use super::youtu_vl_lm_config::YoutuTextConfig;

/// Apply the language-side sanitizations that Python `Model.sanitize` performs:
///
/// 1. `kv_b_proj.weight` is decomposed per-head into `embed_q.weight` (the
///    nope half, transposed) and `unembed_out.weight` (the v half).
/// 2. When `tie_word_embeddings` is true, drop any `lm_head.weight` entry
///    that the safetensors archive may still carry.
///
/// The caller is expected to have already stripped the `language_model.`
/// prefix and the `siglip2.vision_model.` / `merger.` remappings should have
/// happened on the vision side.
///
/// Returns `Err` if a quantized `kv_b_proj` layer is missing its `biases`
/// tensor, or if the resulting weight shape is inconsistent with the config.
///
/// Used by: `loading::vlm_youtu_vl::load_youtu_vl_vlm`.
pub fn sanitize_text_weights(
    mut weights: WeightMap,
    config: &YoutuTextConfig,
) -> Result<WeightMap, String> {
    let num_heads = config.num_attention_heads as i32;
    let head_dim = (config.qk_nope_head_dim + config.v_head_dim) as i32;
    let qk_nope_head_dim = config.qk_nope_head_dim as i32;
    let kv_lora_rank = config.kv_lora_rank as i32;

    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        let kv_b_key = format!("{}.kv_b_proj.weight", prefix);
        let embed_q_key = format!("{}.embed_q.weight", prefix);

        if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
            continue;
        }

        let scales_key = format!("{}.kv_b_proj.scales", prefix);
        let is_quantized = weights.contains_key(&scales_key);

        // Take the kv_b_proj weight and (if quantized) dequantize it before
        // splitting per-head — splitting in quantized space would scramble the
        // per-group scales/biases.
        let w = weights.remove(&kv_b_key).unwrap();
        let w_full = if is_quantized {
            let s = weights.remove(&scales_key).unwrap();
            // M1: biases are mandatory when scales are present; missing biases
            // indicate a malformed or partially-converted checkpoint.
            let b_key = format!("{}.kv_b_proj.biases", prefix);
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

        // M2: validate that w_full has the expected [num_heads * head_dim, kv_lora_rank]
        // shape before reshaping — a mismatch means the checkpoint config disagrees
        // with the stored tensor dimensions.
        let w_shape = mlxcel_core::array_shape(&w_full);
        let expected_rows = num_heads * head_dim;
        if w_shape.len() != 2 || w_shape[0] != expected_rows || w_shape[1] != kv_lora_rank {
            return Err(format!(
                "layer {layer_idx}: kv_b_proj shape mismatch — got {:?}, expected \
                 [{expected_rows}, {kv_lora_rank}] (num_heads={num_heads}, \
                 head_dim={head_dim}, kv_lora_rank={kv_lora_rank})",
                w_shape
            ));
        }

        // [num_heads * head_dim, kv_lora_rank] → [num_heads, head_dim, kv_lora_rank]
        let w_3d = mlxcel_core::reshape(&w_full, &[num_heads, head_dim, -1]);

        // wk = w[:, :qk_nope, :]   (nope half)
        // wv = w[:, qk_nope:, :]   (v_head half)
        let wk = mlxcel_core::utils::slice_axis(&w_3d, 1, 0, qk_nope_head_dim);
        let wv = mlxcel_core::utils::slice_axis(&w_3d, 1, qk_nope_head_dim, -1);

        // embed_q stores the swapped axes form: [num_heads, kv_lora_rank, qk_nope]
        let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);

        // Make the slices contiguous so MultiLinear's matmul has well-formed
        // strides regardless of the upstream backend.
        let wk = mlxcel_core::copy(&wk);
        let wv = mlxcel_core::copy(&wv);

        weights.insert(format!("{}.embed_q.weight", prefix), wk);
        weights.insert(format!("{}.unembed_out.weight", prefix), wv);
    }

    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
        weights.remove("lm_head.scales");
        weights.remove("lm_head.biases");
    }

    Ok(weights)
}
